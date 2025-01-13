import React, { useEffect, useRef, useState } from 'react';

// Configuration constants
const GRID_SIZE = 128;
const WORKGROUP_SIZE = 8;
const UPDATE_INTERVAL = 1000 / 2; // 2 Hz

const GameOfLife: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const deviceRef = useRef<GPUDevice | null>(null);
  const frameRef = useRef<number>();
  const lastUpdateRef = useRef<number>(0);
  
  const [isSupported, setIsSupported] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [shaderError, setShaderError] = useState<string | null>(null);

  useEffect(() => {
    const initializeWebGPU = async () => {
      try {
        if (!navigator.gpu) {
          throw new Error('WebGPU not supported');
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error('Couldn\'t request WebGPU adapter');
        }

        const device = await adapter.requestDevice();
        deviceRef.current = device;

        const canvas = canvasRef.current;
        if (!canvas) {
          throw new Error('Canvas not found');
        }

        const context = canvas.getContext('webgpu');
        if (!context) {
          throw new Error('Couldn\'t get WebGPU context');
        }

        const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        canvas.width = GRID_SIZE;
        canvas.height = GRID_SIZE;
        context.configure({ 
          device, 
          format: canvasFormat,
          alphaMode: 'premultiplied',
        });

        // Add render shader to convert R32Uint to canvas format
        const renderShader = `
          @group(0) @binding(0) var inputTexture: texture_2d<u32>;
          @group(0) @binding(1) var outputTexture: texture_storage_2d<${canvasFormat}, write>;

          @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
          fn main(@builtin(global_invocation_id) id: vec3u) {
            let dims = textureDimensions(inputTexture);
            if (id.x >= dims.x || id.y >= dims.y) {
              return;
            }

            let cell = textureLoad(inputTexture, vec2u(id.xy), 0).r;
            let color = vec4f(f32(cell), f32(cell), f32(cell), 1.0);
            textureStore(outputTexture, vec2u(id.xy), color);
          }
        `;

        // Compute shader for Game of Life rules (unchanged)
        const computeShader = `
          @group(0) @binding(0) var currentState: texture_2d<u32>;
          @group(0) @binding(1) var nextState: texture_storage_2d<r32uint, write>;

          fn getCell(pos: vec2u) -> u32 {
            let dims = textureDimensions(currentState);
            return textureLoad(currentState, pos, 0).r;
          }

          @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
          fn main(@builtin(global_invocation_id) id: vec3u) {
            let dims = textureDimensions(currentState);
            if (id.x >= dims.x || id.y >= dims.y) {
              return;
            }

            let pos = vec2u(id.xy);
            var neighbors = 0u;
            
            for (var dy: i32 = -1; dy <= 1; dy++) {
              for (var dx: i32 = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) {
                  continue;
                }
                
                let nx = (i32(pos.x) + dx + i32(dims.x)) % i32(dims.x);
                let ny = (i32(pos.y) + dy + i32(dims.y)) % i32(dims.y);
                neighbors += getCell(vec2u(u32(nx), u32(ny)));
              }
            }

            let current = getCell(pos);
            var next = 0u;
            
            if (current == 1u) {
              if (neighbors == 2u || neighbors == 3u) {
                next = 1u;
              }
            } else if (neighbors == 3u) {
              next = 1u;
            }

            textureStore(nextState, pos, vec4u(next));
          }
        `;

        // Create compute pipelines
        const computePipeline = device.createComputePipeline({
          layout: 'auto',
          compute: {
            module: device.createShaderModule({ code: computeShader }),
            entryPoint: 'main',
          },
        });

        const renderPipeline = device.createComputePipeline({
          layout: 'auto',
          compute: {
            module: device.createShaderModule({ code: renderShader }),
            entryPoint: 'main',
          },
        });

        // Create cell state textures (R32Uint format)
        const cellTextures = [0, 1].map(() => 
          device.createTexture({
            size: [GRID_SIZE, GRID_SIZE],
            format: 'r32uint',
            usage: 
              GPUTextureUsage.COPY_DST |
              GPUTextureUsage.STORAGE_BINDING |
              GPUTextureUsage.TEXTURE_BINDING,
          })
        );

        // Create output texture for rendering (canvas format)
        const outputTexture = device.createTexture({
          size: [GRID_SIZE, GRID_SIZE],
          format: canvasFormat,
          usage: 
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.COPY_SRC,
        });

        // Initialize with random data
        const initialData = new Uint32Array(GRID_SIZE * GRID_SIZE);
        for (let i = 0; i < initialData.length; i++) {
          initialData[i] = Math.random() > 0.7 ? 1 : 0;
        }

        const stagingBuffer = device.createBuffer({
          size: initialData.byteLength,
          usage: GPUBufferUsage.COPY_SRC,
          mappedAtCreation: true,
        });

        new Uint32Array(stagingBuffer.getMappedRange()).set(initialData);
        stagingBuffer.unmap();

        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToTexture(
          { 
            buffer: stagingBuffer,
            bytesPerRow: GRID_SIZE * 4,
          },
          { texture: cellTextures[0] },
          { width: GRID_SIZE, height: GRID_SIZE }
        );

        device.queue.submit([commandEncoder.finish()]);

        // Create bind groups
        const computeBindGroups = [0, 1].map((i, idx) => {
          const nextIdx = (idx + 1) % 2;
          return device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: cellTextures[idx].createView() },
              { binding: 1, resource: cellTextures[nextIdx].createView() },
            ],
          });
        });

        const renderBindGroups = [0, 1].map((_, idx) => 
          device.createBindGroup({
            layout: renderPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: cellTextures[idx].createView() },
              { binding: 1, resource: outputTexture.createView() },
            ],
          })
        );

        let currentTexture = 0;

        const updateSimulation = (timestamp: number) => {
          if (isPaused) {
            frameRef.current = requestAnimationFrame(updateSimulation);
            return;
          }

          const deltaTime = timestamp - lastUpdateRef.current;

          if (deltaTime >= UPDATE_INTERVAL) {
            const commandEncoder = device.createCommandEncoder();
            
            // Compute pass to update state
            const computePass = commandEncoder.beginComputePass();
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, computeBindGroups[currentTexture]);
            computePass.dispatchWorkgroups(
              Math.ceil(GRID_SIZE / WORKGROUP_SIZE),
              Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
            );
            computePass.end();

            // Render pass to convert format
            const renderPass = commandEncoder.beginComputePass();
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, renderBindGroups[currentTexture]);
            renderPass.dispatchWorkgroups(
              Math.ceil(GRID_SIZE / WORKGROUP_SIZE),
              Math.ceil(GRID_SIZE / WORKGROUP_SIZE)
            );
            renderPass.end();

            // Copy to canvas
            commandEncoder.copyTextureToTexture(
              { texture: outputTexture },
              { texture: context.getCurrentTexture() },
              { width: GRID_SIZE, height: GRID_SIZE }
            );

            device.queue.submit([commandEncoder.finish()]);

            currentTexture = (currentTexture + 1) % 2;
            lastUpdateRef.current = timestamp;
            setGeneration(prev => prev + 1);
          }

          frameRef.current = requestAnimationFrame(updateSimulation);
        };

        frameRef.current = requestAnimationFrame(updateSimulation);

      } catch (error) {
        console.error('WebGPU initialization failed:', error);
        setShaderError(error.message);
        setIsSupported(false);
      }
    };

    initializeWebGPU();

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
      if (deviceRef.current) {
        deviceRef.current.destroy();
      }
    };
  }, [isPaused]);

  // Rest of the component remains the same
  if (shaderError) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
        <h2 className="text-xl mb-4">Failed to load shader</h2>
        <p className="text-red-400">{shaderError}</p>
      </div>
    );
  }

  if (!isSupported) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
        <h2 className="text-xl mb-4">WebGPU is not supported in your browser</h2>
        <p>Please try Chrome Canary or another WebGPU-enabled browser.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 p-8">
      <div className="mb-4 flex gap-4">
        <button
          onClick={() => setIsPaused(!isPaused)}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
        >
          {isPaused ? 'Resume' : 'Pause'}
        </button>
        <button
          onClick={() => setGeneration(0)}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
        >
          Reset
        </button>
      </div>
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="border border-gray-600"
          style={{
            width: `${GRID_SIZE * 4}px`,
            height: `${GRID_SIZE * 4}px`,
            imageRendering: 'pixelated',
          }}
        />
        <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded">
          Generation: {generation}
        </div>
      </div>
    </div>
  );
};

export default GameOfLife;