# TinyLlama
touch TinyLlama-1.1B-Chat-v1.0.txt
for i in {1..10}; do
    echo "TinyLlama $i"
    python compile_test.py ../genAI/models/TinyLlama-1.1B-Chat-v1.0
done

# Llama-2-7b
touch Llama-2-7b-hf.txt
for i in {1..10}; do
    echo "Llama-2-7b $i"
    python compile_test.py ../genAI/models/Llama-2-7b-hf
done

# Llama-3-8B
touch Meta-Llama-3-8B-Instruct.txt
for i in {1..10}; do
    echo "Llama-3-8B $i"
    python compile_test.py ../genAI/models/Meta-Llama-3-8B-Instruct
done