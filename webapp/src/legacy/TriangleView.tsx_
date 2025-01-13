import React, { useEffect, useRef } from 'react';

const AIView = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const resizeCanvas = () => {
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
      }
    };

    const initializeWebGPU = async () => {
      if (!navigator.gpu) {
        console.error('WebGPU is not supported on this browser.');
        return;
      }

      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();
      const canvas = canvasRef.current;
      const context = canvas.getContext('webgpu');

      const swapChainFormat = 'bgra8unorm';
      context.configure({
        device,
        format: swapChainFormat,
      });

      // Define the render pipeline with corrected WGSL syntax
      const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: device.createShaderModule({
            code: `
            @vertex
            fn main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
              var positions = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 0.5),
                vec2<f32>(-0.5, -0.5),
                vec2<f32>(0.5, -0.5)
              );
              let position = positions[VertexIndex];
              return vec4<f32>(position, 0.0, 1.0);
            }
            `,
          }),
          entryPoint: 'main',
        },
        fragment: {
          module: device.createShaderModule({
            code: `
            @fragment
            fn main() -> @location(0) vec4<f32> {
              return vec4<f32>(1.0, 0.0, 0.0, 1.0);
            }
            `,
          }),
          entryPoint: 'main',
          targets: [{ format: swapChainFormat }],
        },
        primitive: { topology: 'triangle-list' },
      });

      const render = () => {
        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
          colorAttachments: [{
            view: textureView,
            clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          }],
        });
        renderPass.setPipeline(pipeline);
        renderPass.draw(3, 1, 0, 0);
        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
      };

      resizeCanvas(); // Set initial canvas size
      render();

      // Redraw whenever the canvas is resized
      window.addEventListener('resize', resizeCanvas);
      return () => window.removeEventListener('resize', resizeCanvas);
    };

    initializeWebGPU();
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
      }}
    />
  );
};

export default AIView;
