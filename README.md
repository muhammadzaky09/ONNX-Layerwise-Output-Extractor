## Prerequisites
```pip install onnx numpy onnxruntime```

## Structures
```parser.py ``` generates layer-wise JSON information on the model

```extractor.py``` modifies the ONNX model to add layer-wise output

## Sample Output 
```{'Conv_0': array([[-4.6328645, -3.281672 , -2.3358371, -1.3900023, -3.1465528,
        -3.1465528, -5.0382223, -0.9846446, -2.4709566, -2.8763142]],
      dtype=float32)} ```

