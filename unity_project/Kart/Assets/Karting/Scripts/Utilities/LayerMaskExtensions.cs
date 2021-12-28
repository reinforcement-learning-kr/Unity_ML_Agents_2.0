using UnityEngine;

public static class LayerMaskExtensions
{
    /// <summary>
    /// Checks whether a LayerMask contains the layer that the given gameobject is on.
    /// </summary>
    public static bool ContainsGameObjectsLayer (this LayerMask layerMask, GameObject gameObject)
    {
        return layerMask == (layerMask | (1 << gameObject.layer));
    }
}
