using System.Collections.Generic;
using UnityEngine;

public static class DictionaryExtensions
{
    /// <summary>
    /// Checks to see if a dictionary contains a specific key and value together.
    /// </summary>
    /// <param name="dictionary">The dictionary to check.</param>
    /// <param name="key">The key part of the pair to check the dictionary for.</param>
    /// <param name="value">The value part of the pair to check the dictionary for.</param>
    /// <returns>Returns true if the dictionary contains the key and that key has the given value.  Otherwise returns false.</returns>
    public static bool ContainsKeyValuePair<TKey, TValue> (this Dictionary<TKey, TValue> dictionary, TKey key, TValue value)
    {
        return dictionary.TryGetValue (key, out TValue outValue) && value.Equals(outValue);
    }
}
