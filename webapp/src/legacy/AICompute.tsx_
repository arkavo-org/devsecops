import { useEffect } from 'react';

const AICompute = () => {
  useEffect(() => {
    const initializeWebGPU = async () => {
      if (!navigator.gpu) {
        console.error('WebGPU is not supported on this browser.');
        return;
      }

      const adapter = await navigator.gpu.requestAdapter();
      const device = await adapter.requestDevice();

      // Define the 128x128x128 array size and total number of elements
      const size = 128 * 128 * 128;

      // Create a buffer to hold the input data (array of 1s as integers)
      const inputBuffer = device.createBuffer({
        size: size * 4, // 4 bytes per int32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      // Initialize the buffer with 1s as integers
      new Int32Array(inputBuffer.getMappedRange()).fill(1);
      inputBuffer.unmap();

      // Create a buffer to store the result as an integer
      const resultBuffer = device.createBuffer({
        size: 4, // 4 bytes for a single int32 result
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      // Initialize result buffer to zero
      new Int32Array(resultBuffer.getMappedRange())[0] = 0;
      resultBuffer.unmap();

      // Create a compute shader that adds up the elements
      const shaderModule = device.createShaderModule({
        code: `
        @group(0) @binding(0) var<storage, read> inputArray: array<i32>;
        @group(0) @binding(1) var<storage, read_write> result: atomic<i32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          if (index < ${size}u) {
            atomicAdd(&result, inputArray[index]);
          }
        }
        `,
      });

      // Set up the compute pipeline
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });

      // Set up the bind group to pass the buffers to the shader
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: resultBuffer } },
        ],
      });

      // Encode and submit the compute pass
      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(size / 64));
      passEncoder.end();

      // Copy the result to a readable buffer
      const readBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      commandEncoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, 4);

      device.queue.submit([commandEncoder.finish()]);

      // Read and log the result
      await readBuffer.mapAsync(GPUMapMode.READ);
      const resultArray = new Int32Array(readBuffer.getMappedRange());
      console.log('Sum of elements:', resultArray[0]);
      readBuffer.unmap();
    };

    initializeWebGPU();
  }, []);

  return <div>Compute Shader Example Running in Console</div>;
};

export default AICompute;
