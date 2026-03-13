# GFN V5 Pooling Plugin Architecture

Las antiguas clases de agregación han sido reformateadas como **Pooling Plugins** de primer nivel, integrándose directamente en el ciclo de vida del `BaseModel` mediante **Hooks**.

## Jerarquía de Componentes
1.  **Motor de Pooling**: (`HamiltonianPooling`, `HierarchicalAggregator`, `MomentumAggregator`)
    *   Ubicación: `gfn/models/components/pooling/`
    *   Función: Reducción de secuencia $[B, L, D] \to [B, D]$.
2.  **PoolingPlugin**:
    *   Ubicación: `gfn/models/components/pooling/__init__.py`
    *   Función: Captura la trayectoria $(x, v)$ reactivamente y dispara el motor al final del batch.

## Cómo Usarlo
```python
model = gfn.create('manifold', pooling_type='hamiltonian')
logits, (x, v), info = model(input_ids)

# El resultado del pooling está en info["plugin_results"]
pooled_state = info["plugin_results"][-1][0] 
```

## Beneficios
- **Memoria Eficiente**: La trayectoria se limpia automáticamente en cada `on_batch_start`.
- **Desacoplamiento**: El modelo core no sabe nada de pooling; el plugin "escucha" la evolución y genera el resumen.
