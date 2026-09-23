// get mlx active compute encoder for the current stream 
auto& compute_encoder = d.get_comand_encoder(s.index);

// 2 set the compiled mls pipeline
compute_encoder.set_compute_pipeline_state(kernel);

// 3. Bind to the argument table (buffer(0),..)
compute_encoder.set_input_array(q, 0) // maps to [[buffer(0)]]
compute_encoder.set_input_array(k, 1) // maps to [[buffer(1)]]
compute_encoder.set_input_array(v, 2) // maps to [[buffer(2)]]
compute_encoder.set_input_array(out, 3) // maps to [[buffer(3)]]

// 4. Pass scalar hyperparameter via setBytes
compute_encoder.set_bytes(scale, sizeof(float), 4); // maps to buffer(4)

// 5. dispatch grid & threadgroups
compute_encoder.dispatch_threadgroups(grid_dimsm group_dims);