using System.Collections.Generic;
using UnityEngine;

public static class ListExtensions
{
    /// <summary>
    /// Gets the next element in a list including cycling back to the 0th element if at the end of the list.
    /// </summary>
    /// <returns>The next element in the cycle.</returns>
    public static TElement GetNextInCycle<TElement> (this List<TElement> list, TElement element)
    {
        int index = list.IndexOf (element);
        index = (index + 1) % list.Count;
        return list[index];
    }
}
