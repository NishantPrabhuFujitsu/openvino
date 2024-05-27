// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <chrono>

namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
	    return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};
}

int main(int argc, char* argv[]) try {
    if (argc != 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }
    // Compile models
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt

    //Read the tokenizer model information from the file to later get the runtime information
    std::cout << "[INFO] Loading tokenizer" << std::endl;
    auto tokenizer_model = core.read_model(std::string{argv[1]} + "/openvino_tokenizer.xml");

    // tokenizer and detokenizer work on CPU only
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    auto [input_ids, attention_mask] = tokenize(tokenizer, argv[2]);
    ov::InferRequest detokenizer = core.compile_model(std::string{argv[1]} + "/openvino_detokenizer.xml", "CPU").create_infer_request();

    // The model can be compiled for GPU as well
    std::cout << "[INFO] Loading model" << std::endl;
    ov::CompiledModel compiled_model = core.compile_model(
        std::string{argv[1]} + "/openvino_model.xml", 
        "CPU", ov::device::properties("CPU", ov::enable_profiling(true))
    );
    ov::InferRequest lm = compiled_model.create_infer_request();
    auto seq_len = input_ids.get_size();
    
    // Initialize inputs
    std::cout << "[INFO] Initializing inference" << std::endl;
    lm.set_tensor("input_ids", input_ids);
    lm.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = lm.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + seq_len, 0);
    constexpr size_t BATCH_SIZE = 1;
    // Input values are persistent between inference calls.
    // That allows to set values, which aren't going to change, only once
    lm.get_tensor("beam_idx").set_shape({BATCH_SIZE});
    lm.get_tensor("beam_idx").data<int32_t>()[0] = 0;
    lm.infer();
    size_t vocab_size = lm.get_tensor("logits").get_shape().back();
    float* logits = lm.get_tensor("logits").data<float>() + (seq_len - 1) * vocab_size;
    int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

    lm.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    position_ids.set_shape({BATCH_SIZE, 1});
    TextStreamer text_streamer{std::move(detokenizer)};

    // Get the runtime info from the tokenizer model that we read earlier
    auto rt_info = tokenizer_model->get_rt_info(); //Get the runtime info for the model
    int64_t SPECIAL_EOS_TOKEN;

    if (rt_info.count("eos_token_id") > 0) { //check if the runtime information has a valid EOS token ID
        SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();
    } else {
        throw std::runtime_error("EOS token ID not found in model's runtime information.");
    }

    std::cout << "[INFO] Starting inference" << std::endl;
    int max_sequence_length = 10;
    double total_time = 0.0;
    unsigned int total_tokens = 0;

    while (out_token != SPECIAL_EOS_TOKEN && seq_len < max_sequence_length) {
        ++seq_len;
        lm.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        lm.get_tensor("attention_mask").set_shape({BATCH_SIZE, seq_len});
        std::fill_n(lm.get_tensor("attention_mask").data<int64_t>(), seq_len, 1);
        position_ids.data<int64_t>()[0] = int64_t(seq_len - 1);
        auto start = std::chrono::high_resolution_clock::now();
        lm.start_async();
        text_streamer.put(out_token);
        lm.wait();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        total_time += (double)duration.count() / 1000.0; 
        total_tokens += 1;
        logits = lm.get_tensor("logits").data<float>();
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }
    text_streamer.end();
    // Model is stateful which means that context (kv-cache) which belongs to a particular
    // text sequence is accumulated inside the model during the generation loop above.
    // This context should be reset before processing the next text sequence.
    // While it is not required to reset context in this sample as only one sequence is processed,
    // it is called for education purposes:
    lm.reset_state();

    std::string FLAGS_exec_graph_path = "greedy_causal_lm.exec_graph.xml";
    try {
        ov::serialize(compiled_model.get_runtime_model(), FLAGS_exec_graph_path);
        std::cerr << "Executable graph is stored to " << FLAGS_exec_graph_path << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Can't get executable graph: " << ex.what() << std::endl;
    }

    std::cout << "=============================================" << std::endl;
    std::cout << "* Total tokens        : " << total_tokens << std::endl;
    std::cout << "* Total time (sec)    : " << total_time << std::endl;
    std::cout << "* Tokens / sec        : " << total_tokens / total_time << std::endl;

} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}