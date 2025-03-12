use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn print_tensor_info(tensors: &SafeTensors) {
        println!("Available tensors:");
        for (name, tensor) in tensors.tensors() {
            println!("- {}: shape={:?}, dtype={:?}", 
                name, 
                tensor.shape(),
                tensor.dtype()
            );
        }
    }

    pub fn from_safetensors(tensors: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 打印张量信息以便调试
        Self::print_tensor_info(tensors);
        
        let n_layers = config.num_hidden_layers;
        
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = tensors.tensor(name).unwrap_or_else(|_| 
                panic!("Failed to find tensor: {}", name));
            let shape = tensor.shape().to_vec();
            let bytes = tensor.data();
            
            let data: Vec<f32> = bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            
            Tensor::new(data, &shape)
        };

        // 注意：lm_head.weight 同时用作 embedding 和 output 层
        let shared_weights = get_tensor("lm_head.weight");
        let embedding_table = shared_weights.clone();
        let lm_head = shared_weights;

        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            // 使用实际的张量名称
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)));
        }

        let rms_out_w = get_tensor("model.norm.weight");

        Self {
            embedding_table,
            lm_head,
            rms_att_w,
            rms_ffn_w,
            rms_out_w,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
        }
    }
}
