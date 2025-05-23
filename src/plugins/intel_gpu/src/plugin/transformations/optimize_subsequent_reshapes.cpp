// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "optimize_subsequent_reshapes.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {

OptimizeSubsequentReshapes::OptimizeSubsequentReshapes() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    auto single_dynamic_dim = [](Output<Node> output) {
        const auto& shape = output.get_partial_shape();

        if (shape.rank().is_dynamic())
            return false;

        if (shape.size() <= 1)
            return false;

        auto dynamic_dims = 0;
        for (size_t i = 0; i < shape.size(); i++)
            dynamic_dims += shape[i].is_dynamic() ? 1 : 0;

        if (dynamic_dims != 1)
            return false;

        return true;
    };

    auto first_reshape_data = any_input(single_dynamic_dim);
    auto first_reshape_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto first_reshape = wrap_type<ov::op::v1::Reshape>({ first_reshape_data, first_reshape_pattern },
                                                        single_dynamic_dim && ov::pass::pattern::consumers_count(1));

    auto second_reshape_pattern = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto second_reshape = wrap_type<ov::op::v1::Reshape>({ first_reshape, second_reshape_pattern }, single_dynamic_dim);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto input_node = pattern_map.at(first_reshape_data).get_node_shared_ptr();
        auto first_reshape_node = pattern_map.at(first_reshape).get_node_shared_ptr();
        auto second_reshape_node = pattern_map.at(second_reshape).get_node_shared_ptr();

        auto input_ps = first_reshape_node->input(0).get_partial_shape();
        auto first_reshape_ps = first_reshape_node->get_output_partial_shape(0);
        auto second_reshape_ps = second_reshape_node->get_output_partial_shape(0);

        auto static_dims_product = [](ov::PartialShape& ps) {
            int64_t total_dims = 1;

            for (auto& dim : ps) {
                if (dim.is_static())
                    total_dims *= dim.get_length();
            }

            return total_dims;
        };

        if (static_dims_product(input_ps) != static_dims_product(first_reshape_ps) ||
            static_dims_product(first_reshape_ps) != static_dims_product(second_reshape_ps))
            return false;

        std::vector<int32_t> new_pattern;
        for (auto& dim : second_reshape_ps) {
            if (dim.is_dynamic()) {
                new_pattern.push_back(-1);
            } else {
                new_pattern.push_back(dim.get_length());
            }
        }

        auto new_pattern_const = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{new_pattern.size()}, new_pattern);
        auto new_reshape = std::make_shared<ov::op::v1::Reshape>(first_reshape_node->input(0).get_source_output(), new_pattern_const, false);
        new_reshape->set_friendly_name(second_reshape_node->get_friendly_name());

        ov::replace_node(second_reshape_node, new_reshape);
        copy_runtime_info(first_reshape_node, new_reshape);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(second_reshape, "OptimizeSubsequentReshapes");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
