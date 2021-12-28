using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class QuaternionExtensions
{
    /// <summary>
    /// Scales a quaternion (including the w component) by a float.  Used for weighting rotation collections.  WARNING: The returned quaternion is NOT normalised.
    /// Note that in order to get a weighted average for quaternions, they should be individually scaled, added and then the result should be normalised.
    /// When adding weighted quaternions make sure their dot product is positive, if not scale the weighted quaternion by -1.
    /// </summary>
    /// <returns>Returns the scaled non-normalised quaternion.</returns>
    public static Quaternion Scale (this Quaternion quaternion, float scale)
    {
        return new Quaternion(quaternion.x * scale, quaternion.y * scale, quaternion.z * scale, quaternion.w * scale);
    }

    /// <summary>
    /// Adds one quaternion to another component-wise.  Used for summing weighted rotation collections.  WARNING: The returned quaternion is NOT normalised.
    /// Note that in order to get a weighted average for quaternions, they should be individually scaled, added and then the result should be normalised.
    /// When adding weighted quaternions make sure their dot product is positive, if not scale the weighted quaternion by -1.
    /// </summary>
    /// <returns>Returns the a non-normalised quaternion with added components.</returns>
    public static Quaternion Add (this Quaternion quaternion, Quaternion toAdd)
    {
        return new Quaternion(quaternion.x + toAdd.x, quaternion.y + toAdd.y, quaternion.z + toAdd.z, quaternion.w + toAdd.w);
    }
}
