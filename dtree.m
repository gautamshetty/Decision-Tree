function [ output_args ] = dtree( training_file, test_file, option, pruning_thr )
    
    file_data = load (training_file);
    [r, c] = size(file_data);
    
    attributes = [1:c];
    
    tree_id = 0;
    tree_array = [];
    if option == "optimized"
        tree = DTL(file_data, attributes, -1, pruning_thr, 1, 'optimized', tree_id);
        tree_array = [tree_array tree];
    elseif option == "randomized"
        tree = DTL(file_data, attributes, -1, pruning_thr, 1, 'randomized', tree_id);
        tree_array = [tree_array tree];
    elseif option == "forest3" || option == "forest15"
        no_of_trees = 3;
        if option == "forest15"
            no_of_trees = 15;
        end
        
        for i=1:no_of_trees
            tree = DTL(file_data, attributes, distribution(file_data), pruning_thr, 1, 'randomized', tree_id);
            tree_array = [tree_array tree];
            tree_id = tree_id + 1;
            disp("\n");
        end
    end
    
    test_data = load (test_file);
    
    classify(tree_array, test_data);
    
end

function [] = classify(tree_array, test_data)

    [r, c] = size(test_data);
    cum_accuracy = 0;
    for i=1:r
        
        record = test_data (i, :);
        
        if (length(tree_array) > 1)
            avg_node = get_empty_node();
            avg_node(1).default(1, :) = zeros(1, 1);
            avg_node(1).default(2, :) = zeros(1, 1);
            avg_cl = avg_node(1).default(1, :);
            avg_pr = avg_node(1).default(2, :);
            for j=1:length(tree_array)

                tree = tree_array(1, j);
                node = get_probability_matrix(tree, record);

                p_array = node(1).default(2, :);
                c_array = node(1).default(1, :);
                
                for k=1:length(c_array)
                    cl = c_array(1, k);
                    pr = p_array(1, k);
                    if j == 1 && k == 1
                        avg_cl(1, 1) = cl;
                        avg_pr(1, 1) = pr;
                    else
                        index = find(avg_cl == cl);
                        if (isempty(index))
                            avg_cl = [avg_cl cl];
                            avg_pr = [avg_pr pr];
                        else
                            avg_pr(1, index) = avg_pr(1, index) + pr;
                        end
                    end
                end
            end
            
            avg_pr = avg_pr / length(tree_array);
            class_array = avg_cl;
            prob_array = avg_pr;
        else
            tree = tree_array(1, 1);
            node = get_probability_matrix(tree, record);
            class_array = node(1).default(1, :);
            prob_array = node(1).default(2, :);
        end
        
        [max_value, max_prob_index] = max(prob_array);
        predicted_class = class_array(1, max_prob_index);
        t = record(:, c);
        
        accuracy = 0;
        all_max = find (prob_array == max_value);
        if (length(all_max) > 1)
            found = false;
            for j=1:length(all_max)
                if class_array(:, all_max(:,j)) == t
                    found = true;
                    predicted_class = class_array(:, all_max(:,j));
                    break;
                end
            end
            
            if found == 1
                accuracy = 1/(length(all_max));
            end
        else
            if class_array(:, max_prob_index) == t
                accuracy = 1;
            end
        end
        
        cum_accuracy = cum_accuracy + accuracy;
        fprintf("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n", (i - 1), predicted_class, t, accuracy);        
    end
    
    fprintf("classification accuracy=%6.4f\n", (cum_accuracy/r));
end

function [node] = get_probability_matrix(tree, record)

    node = tree;
    while node(1).threshold ~= -1
        
        value = record(1, node(1).attribute);
        if (value < node(1).threshold)
            node = node(1).left_child;
        else
            node = node(1).right_child;
        end
    end
    
end

function [tree] = DTL(examples, attributes, default, pruning_thr, node_id, option, tree_id)
    
    [er, ec] = size(examples);
    if isempty(examples)
        tree = get_empty_node();
        tree(1).attribute = -1;
        tree(1).threshold = -1;
        tree(1).gain = -1;
        tree(1).node_id = node_id;
        tree(1).default = default;
    elseif er < pruning_thr
        %tree = get_node(default, -1, node_id, -1);
        tree = get_empty_node();
        tree(1).attribute = -1;
        tree(1).threshold = -1;
        tree(1).gain = -1;
        tree(1).node_id = node_id;
        tree(1).default = default;
    else
        if length(unique(examples(:,length(attributes)))) == 1
            %tree = get_node(ec, -1, node_id, -1);
            tree = get_empty_node();
            tree(1).attribute = -1;
            tree(1).threshold = -1;
            tree(1).gain = -1;
            tree(1).node_id = node_id;
            tree(1).default = default;
        else
            if option == "optimized"
                [best_attribute, best_threshold, gain] = choose_attribute(examples, attributes);
            else 
                [best_attribute, best_threshold, gain] = choose_random_attribute(examples, attributes);
            end
            
            %tree = get_node(best_attribute, best_threshold, node_id, gain);
            tree = get_empty_node();
            tree(1).attribute = best_attribute;
            tree(1).threshold = best_threshold;
            tree(1).node_id = node_id;
            tree(1).gain = gain;
            
            [r, c] = size(examples);
            
            examples_left = sub_matrix(examples, find (examples(:,best_attribute) < best_threshold), c);
            examples_right = sub_matrix(examples, find (examples(:,best_attribute) >= best_threshold), c);
            
            tree(1).left_child.node_id = 2 * tree(1).node_id;
            tree(1).left_child = DTL(examples_left, attributes, distribution(examples_left), pruning_thr, tree(1).left_child.node_id, option, tree_id);
            
            tree(1).right_child.node_id = 2 * tree(1).node_id + 1;
            tree(1).right_child = DTL(examples_right, attributes, distribution(examples_right), pruning_thr, tree(1).right_child.node_id, option, tree_id);
            
        end
    end
    
    feature = tree(1).attribute;
    if tree(1).attribute > 0
        feature = feature - 1;
    end
    fprintf("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n", tree_id, tree(1).node_id, feature, tree(1).threshold, tree(1).gain);
    
end

function [prob_mat] = distribution(examples)

    [r, c] = size(examples);
    classes = unique(examples(:, c));
    num_each_class = histc(examples(:,c), classes);
    total_classes = sum(num_each_class);
    prob_array = num_each_class/total_classes;
    
    prob_mat = zeros(2, length(classes));
    prob_mat(1, :) = classes;
    prob_mat(2, :) = prob_array;
end

function [sub_examples] = sub_matrix(examples, row_indices, column_size)
    
    index = 1;
    sub_examples = zeros(length(row_indices), column_size);    
    for i=1:length(row_indices)
        row_index = row_indices(index);
        sub_examples(i,:) = examples(row_index, :);
        index = index + 1;
    end
end

function [node] = get_empty_node()
    node = struct('node_id', {}, 'attribute', {}, 'threshold', {}, 'info_gain', {}, 'left_child', {}, 'right_child', {}, 'default', {});
end

function [node] = get_node(attribute, threshold, node_id, gain)
    node = struct('node_id', node_id, 'attribute', attribute, 'threshold', threshold, 'info_gain', gain, 'left_child', {}, 'right_child', {}, 'default', {});
end

function [best_attribute, best_threshold, gain] = choose_attribute(examples, attributes)

    max_gain = -1;
    best_attribute = -1;
    best_threshold = -1;    
    for i=1:(length(attributes) - 1)        
        attr = attributes(i);
        attribute_values = examples(:, attr);
        l = min(attribute_values);
        m = max(attribute_values);
        for k=1:50
            threshold = l + k * (m - l)/51;
            gain = information_gain(examples, attr, threshold);
            if gain > max_gain
                max_gain = gain;
                best_attribute = attr;
                best_threshold = threshold;
            end
        end
    end
end

function [best_attribute, best_threshold, gain] = choose_random_attribute(examples, attributes)

    max_gain = -1;
    best_attribute = -1;
    best_threshold = -1;
    attr = randi([1, (length(attributes) - 1)]);
    attribute_values = examples(:, attr);
    l = min(attribute_values);
    m = max(attribute_values);
    for k=1:50
        threshold = l + k * (m - l)/51;
        gain = information_gain(examples, attr, threshold);
        if gain > max_gain
            max_gain = gain;
            best_attribute = attr;
            best_threshold = threshold;
        end
    end

end

function [info_gain] = information_gain(examples, attr, threshold)
    
    [r, c] = size(examples);
    attr_values = examples(:, attr);
    attr_values = [attr_values examples(:,c)];
    [r, c] = size(attr_values);
    classes = unique(attr_values(:, c));
    num_each_class = histc(attr_values(:,c), classes);
    K = sum(num_each_class);
    
    left_children = sub_matrix(attr_values, find (attr_values(:,1) < threshold), c);
    left_classes = unique(left_children(:,c));
    num_left_class = histc(left_children(:,c), left_classes);
    
    right_children = sub_matrix(attr_values, find (attr_values(:,1) >= threshold), c);
    right_classes = unique(right_children(:,c));
    num_right_class = histc(right_children(:,c), right_classes);
    
    info_gain = entropy(num_each_class) - (sum(num_left_class)/K) * entropy(num_left_class) - (sum(num_right_class)/K) * entropy(num_right_class);
end

function [entropy] = entropy(node_values)
    
    k = sum(node_values);    
    entropy = 0;
    for i=1:length(node_values)
        entropy = entropy - (node_values(i)/k) * log2(node_values(i)/k);
    end
    
end
