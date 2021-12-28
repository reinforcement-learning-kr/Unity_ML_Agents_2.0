using System;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

/// <summary>
/// A drawer for an attribute that forces a public field to implement an interface.
/// </summary>
[CustomPropertyDrawer(typeof (RequireInterfaceAttribute))]
sealed class RequireInterfaceAttributeDrawer : PropertyDrawer
{
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        if (property.propertyType != SerializedPropertyType.ObjectReference)
        {
            Debug.LogError("RequireInterfaceAttribute can only be used on fields of type UnityEngine.Object or one of its subclasses!");
            return;
        }
        
        Object propertyObject = property.objectReferenceValue;
        if (propertyObject != null)
        {
            Type interfaceType = ((RequireInterfaceAttribute)attribute).type;
            Type propertyType = propertyObject.GetType ();
            
            if (propertyType.IsAssignableFrom (interfaceType))
            {
                Debug.LogError (propertyObject + " does not implement  " + interfaceType.Name);
                property.objectReferenceValue = null;
            }
        }
        
        EditorGUI.PropertyField(position, property, label);

        if (property.objectReferenceValue is GameObject)
            property.objectReferenceValue = ((GameObject)property.objectReferenceValue).GetComponent (((RequireInterfaceAttribute)attribute).type);
    }
}