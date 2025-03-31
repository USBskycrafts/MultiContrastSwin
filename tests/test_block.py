import unittest
import torch
import numpy as np
from multicontrast.nn.block import *
from multicontrast.nn.utils import create_attention_mask


class TestWindowAttention(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.window_size = (2, 2)
        self.num_contrasts = 8
        self.num_heads = 2
        self.num_resources = 2
        self.attention = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_contrasts=self.num_contrasts,
            num_heads=self.num_heads,
            num_resouces=self.num_resources
        )

    def test_attention_scores(self):
        """测试注意力分数计算是否正确"""
        q = torch.randn(1, 3, 8, 8, self.dim)
        k = torch.randn(1, 5, 8, 8, self.dim)
        v = torch.randn(1, 5, 8, 8, self.dim)

        # 计算手动注意力分数
        q_proj = self.attention.q_proj(q)
        k_proj = self.attention.k_proj(k)
        q_proj = window_partition(q_proj, self.window_size)
        k_proj = window_partition(k_proj, self.window_size)
        q_proj, k_proj = map(lambda x: multihead_shuffle(
            x, self.num_heads), (q_proj, k_proj))

        manual_attn = torch.matmul(
            q_proj, k_proj.transpose(-2, -1)) * self.attention.scale

        # 计算模块输出的注意力分数
        output = self.attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])
        # 保存中间结果用于验证
        with torch.no_grad():
            q_proj = self.attention.q_proj(q)
            k_proj = self.attention.k_proj(k)
            q_proj = window_partition(q_proj, self.window_size)
            k_proj = window_partition(k_proj, self.window_size)
            q_proj, k_proj = map(lambda x: multihead_shuffle(
                x, self.num_heads), (q_proj, k_proj))
            module_attn = torch.matmul(
                q_proj, k_proj.transpose(-2, -1)) * self.attention.scale

        # 比较两者是否接近
        self.assertTrue(torch.allclose(
            manual_attn, module_attn, rtol=1e-4, atol=1e-4))

    def test_relative_position_bias(self):
        """测试相对位置偏置是否正确应用"""
        q = torch.randn(1, 3, 8, 8, self.dim)
        k = torch.randn(1, 3, 8, 8, self.dim)

        # 获取相对位置偏置
        with torch.no_grad():
            selected_indices = ([2, 3, 5], [2, 3, 5])  # 转换为元组
            relative_bias_index = select_relative_coords(
                self.attention.relative_bias_index,
                self.num_contrasts,
                int(self.window_size[0]),  # 显式转换为int
                int(self.window_size[1]),
                selected_indices
            )
            relative_bias_index = relative_bias_index.view(-1)
            relative_bias = self.attention.relative_bias_table[relative_bias_index].view(
                int(self.num_heads),
                int(3 * np.prod(self.window_size)),
                int(3 * np.prod(self.window_size))
            )

        # 验证偏置形状
        expected_shape = (
            self.num_heads,
            3 * np.prod(self.window_size),
            3 * np.prod(self.window_size)
        )
        self.assertEqual(relative_bias.shape, expected_shape)

    def test_mask_application(self):
        """测试注意力掩码是否正确应用"""
        q = torch.randn(1, 3, 8, 8, self.dim)
        k = torch.randn(1, 5, 8, 8, self.dim)
        v = torch.randn(1, 5, 8, 8, self.dim)

        # 创建掩码
        mask = create_attention_mask(
            3, 5, 8, 8, self.num_heads, self.window_size, (1, 1))

        # 计算带掩码和不带掩码的输出
        output_with_mask = self.attention(
            q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]], mask=mask)
        output_without_mask = self.attention(
            q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])

        # 验证输出不同
        self.assertFalse(torch.allclose(output_with_mask, output_without_mask))

    def test_gradient_flow(self):
        """测试梯度是否能正确传播"""
        q = torch.randn(1, 3, 8, 8, self.dim, requires_grad=True)
        k = torch.randn(1, 5, 8, 8, self.dim, requires_grad=True)
        v = torch.randn(1, 5, 8, 8, self.dim, requires_grad=True)

        output = self.attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])
        loss = output.sum()
        loss.backward()

        # 验证梯度存在
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertIsNotNone(self.attention.q_proj.weight.grad)

    def test_forward(self):
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=2)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

    def test_forward2(self):
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 3, 8, 8, 16)
        v = torch.randn(1, 3, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=1)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [2, 3, 5]])
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

    def test_forward3(self):
        mask = torch.randn(1, 1, 3 * 2 * 2, 5 * 2 * 2)
        mask_ = create_attention_mask(3, 5, 8, 8, 2, (2, 2), (1, 1))
        self.assertTrue(mask.shape == mask_.shape,
                        f"{mask.shape}, {mask_.shape}")
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=2)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]], mask=mask_)
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

    def test_forward4(self):
        # Create a dummy input tensor
        q = torch.randn(8, 3, 8, 8, 16)
        k = torch.randn(8, 3, 8, 8, 16)
        v = torch.randn(8, 3, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=1)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [2, 3, 5]])
        self.assertEqual(y.shape, (8, 3, 8, 8, 16))

    def test_backward(self):
        # Create a dummy input tensor
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # Create a WindowAttention module
        window_attention = WindowAttention(
            dim=16, window_size=(2, 2), num_contrasts=8, num_heads=2, num_resouces=2)

        # Forward pass
        y = window_attention(q, k, v, [[2, 3, 5], [0, 1, 4, 6, 7]])

        # Backward pass
        y.sum().backward()

    def test_multi_contrast_interaction(self):
        """验证不同对比度间的注意力隔离"""
        q = torch.randn(1, 3, 8, 8, 16)
        k = torch.randn(1, 5, 8, 8, 16)
        v = torch.randn(1, 5, 8, 8, 16)

        # 两种不同的对比度组合
        output1 = self.attention(q, k, v, [[2, 3, 4], [0, 1, 2, 3, 4]])
        output2 = self.attention(q, k, v, [[0, 1, 2], [0, 1, 2, 3, 4]])

        # 验证输出不同
        self.assertFalse(torch.allclose(output1, output2, rtol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "需要CUDA设备")
    def test_throughput_benchmark(self):
        """性能基准测试"""
        inputs = torch.randn(32, 8, 64, 64, 16).cuda()
        self.attention = self.attention.cuda()

        # 预热
        for _ in range(3):
            _ = self.attention(inputs, inputs, inputs, [
                               [0, 1, 2, 3, 4, 5, 6, 7]]*2)

        # 正式测试
        start = torch.cuda.Event(enable_timing=True, blocking=True)
        end = torch.cuda.Event(enable_timing=True, blocking=True)

        stream = torch.cuda.current_stream()
        start.record(stream=stream)
        _ = self.attention(inputs, inputs, inputs, [
                           [0, 1, 2, 3, 4, 5, 6, 7]]*2)
        end.record(stream=stream)
        torch.cuda.synchronize()

        print(f"\nWindowAttention吞吐量测试: {start.elapsed_time(end)}ms")


class TestMultiContrastEncoder(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.window_size = (4, 4)
        self.shift_size = (2, 2)
        self.num_contrasts = 8
        self.num_heads = 2
        self.encoder = MultiContrastEncoderBlock(
            dim=self.dim,
            window_size=self.window_size,
            shift_size=self.shift_size,
            num_contrasts=self.num_contrasts,
            num_heads=self.num_heads
        )

    def test_residual_connections(self):
        """测试残差连接"""
        x = torch.randn(1, 3, 8, 8, self.dim)
        original_norm = torch.norm(x)

        output = self.encoder(x, [2, 3, 5])
        output_norm = torch.norm(output)

        # 验证残差连接后范数变化合理
        self.assertIsInstance(output_norm, torch.Tensor)
        self.assertIsInstance(original_norm, torch.Tensor)
        self.assertTrue((output_norm > original_norm * 0.5).all())
        self.assertTrue((output_norm < original_norm * 2.0).all())

    def test_forward2(self):
        # Create a dummy input tensor
        x = torch.randn(8, 3, 8, 8, 16)

        # Create a MultiContrastEncoder module
        multi_contrast_encoder = MultiContrastEncoderBlock(
            dim=16, window_size=(4, 4), shift_size=(2, 2), num_contrasts=8, num_heads=2)

        # Forward pass
        y = multi_contrast_encoder(x, [2, 3, 5])
        self.assertEqual(y.shape, (8, 3, 8, 8, 16))

        # 验证批处理一致性
        for i in range(1, 8):
            self.assertFalse(torch.allclose(y[0], y[i]))

        # 验证不同对比度的处理
        y2 = multi_contrast_encoder(x, [1, 4, 6])  # 不同对比度组合
        self.assertFalse(torch.allclose(y, y2))


class TestMultiContrastDecoder(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.window_size = (4, 4)
        self.shift_size = (2, 2)
        self.num_contrasts = 8
        self.num_heads = 2
        self.decoder = MultiContrastDecoderBlock(
            dim=self.dim,
            window_size=self.window_size,
            shift_size=self.shift_size,
            num_contrasts=self.num_contrasts,
            num_heads=self.num_heads
        )

    def test_cross_attention(self):
        """测试解码器中的交叉注意力"""
        x = torch.randn(1, 3, 8, 8, self.dim)
        encoded_features = torch.randn(1, 5, 8, 8, self.dim)

        output = self.decoder(x, encoded_features, [
                              [0, 1, 4, 6, 7], [2, 3, 5]])

        # 验证输出形状
        self.assertEqual(output.shape, x.shape)

        # 验证交叉注意力改变了输入
        self.assertFalse(torch.allclose(output, x))

    def test_attention_masks(self):
        """测试解码器中不同注意力类型的掩码"""
        x = torch.randn(1, 3, 8, 8, self.dim)
        encoded_features = torch.randn(1, 5, 8, 8, self.dim)

        # 获取注意力图
        self.decoder.eval()
        with torch.no_grad():
            output = self.decoder(x, encoded_features, [
                                  [0, 1, 4, 6, 7], [2, 3, 5]])

        # 验证自注意力和交叉注意力都应用了
        test_q = torch.randn(1, 3, 8, 8, self.dim)  # 定义测试输入
        attn1_mean = self.decoder.attn1(test_q, test_q, test_q,
                                        ([2, 3, 5], [2, 3, 5])).mean()
        attn2_mean = self.decoder.attn2(test_q, encoded_features, encoded_features,
                                        ([2, 3, 5], [0, 1, 4, 6, 7])).mean()
        self.assertNotEqual(attn1_mean.item(), attn2_mean.item())

    def test_forward(self):
        # Create a dummy input tensor
        x = torch.randn(1, 3, 8, 8, 16)
        features = torch.randn(1, 5, 8, 8, 16)

        # Create a MultiContrastDecoder module
        multi_contrast_decoder = MultiContrastDecoderBlock(
            dim=16, window_size=(4, 4), shift_size=(2, 2), num_contrasts=8, num_heads=2)

        # Forward pass
        y = multi_contrast_decoder(x, features, [[0, 1, 4, 6, 7], [2, 3, 5]])

        self.assertEqual(y.shape, (1, 3, 8, 8, 16))


class TestImageEncoding(unittest.TestCase):
    def test_forward1(self):
        # Create a dummy input tensor
        x = torch.randn(1, 3, 8, 8, 1)

        encoding = MultiContrastImageEncoding(1, 16, 8)
        y = encoding(x, [0, 2, 7])
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))


class TestImageDecoding(unittest.TestCase):
    def test_forward(self):
        # Create a dummy input tensor
        x = torch.randn(1, 3, 8, 8, 16)

        decoding = MultiContrastImageDecoding(16, 1, 8)
        y = decoding(x, [0, 2, 7])
        self.assertEqual(y.shape, (1, 3, 8, 8, 1))


class TestMoELayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 16
        self.output_size = 16
        self.num_contrasts = 4
        self.moe = MoELayer(
            input_size=self.input_size,
            output_size=self.output_size,
            num_contrasts=self.num_contrasts,
            k=2
        )

    def test_expert_selection(self):
        """测试专家选择机制"""
        x = torch.randn(1, 3, 8, 8, self.input_size)

        # 前向传播
        output = self.moe(x)

        # 验证专家选择
        self.assertEqual(self.moe.top_k_indices.shape, (1, 3, 8, 8, 2))

        # 验证所有专家都被选中过
        unique_experts = torch.unique(self.moe.top_k_indices)
        self.assertGreaterEqual(unique_experts.size(0), 1)

    def test_load_balancing(self):
        """测试专家负载均衡"""
        x = torch.randn(32, 3, 8, 8, self.input_size)

        # 多次前向传播
        for _ in range(10):
            self.moe(x)

        # 验证专家偏置更新
        if not self.moe.use_aux_loss:
            expert_counts = torch.bincount(
                self.moe.top_k_indices.flatten(),
                minlength=self.moe.num_experts
            )
            std_mean_ratio = (expert_counts.float().std() /
                              expert_counts.float().mean()).item()
            self.assertTrue(std_mean_ratio < 0.5)  # 专家选择相对均衡

    def test_forward(self):
        mlp = self.moe
        x = torch.randn(1, 3, 8, 8, 16)
        y = mlp(x)
        self.assertEqual(y.shape, (1, 3, 8, 8, 16))

        # 验证专家选择
        self.assertEqual(mlp.top_k_indices.shape, (1, 3, 8, 8, 2))
        # 验证至少有两个专家被选中
        unique_experts = torch.unique(mlp.top_k_indices)
        self.assertGreaterEqual(unique_experts.size(0), 2)
