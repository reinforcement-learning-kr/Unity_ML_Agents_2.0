using System;
using UnityEngine;

/// <summary>
/// An attribute that forces a public field to implement an interface.
/// </summary>
[AttributeUsage(AttributeTargets.Field)]
public sealed class RequireInterfaceAttribute : PropertyAttribute
{
    public readonly Type type;

    public RequireInterfaceAttribute(Type value)
    {
        if (!value.IsInterface)
        {
            throw new Exception("Type must be an interface!");
        }
        type = value;
    }
}