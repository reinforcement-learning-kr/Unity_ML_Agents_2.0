using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Jiggler : MonoBehaviour
{
    [Range(0, 1)]
    public float power = .1f;

    [Header("Position Jiggler")]
    public bool jigPosition = true;
    public Vector3 positionJigAmount;
    [Range(0, 120)]
    public float positionFrequency = 10;
    float positionTime;

    [Header("Rotation Jiggler")]
    public bool jigRotation = true;
    public Vector3 rotationJigAmount;
    [Range(0, 120)]
    public float rotationFrequency = 10;
    float rotationTime;

    [Header("Scale Jiggler")]
    public bool jigScale = true;
    public Vector3 scaleJigAmount = new Vector3(.1f, -.1f, .1f);
    [Range(0, 120)]
    public float scaleFrequency = 10;
    float scaleTime;

    Vector3 basePosition;
    Quaternion baseRotation;
    Vector3 baseScale;

    void Start(){
        basePosition = this.transform.localPosition;
        baseRotation = this.transform.localRotation;
        baseScale = this.transform.localScale;
    }

    // Update is called once per frame
    void Update()
    {
        var dt = Time.deltaTime;
        positionTime += dt * positionFrequency;
        rotationTime += dt * rotationFrequency;
        scaleTime += dt * scaleFrequency;

        if (jigPosition)
            transform.localPosition = basePosition + positionJigAmount * Mathf.Sin(positionTime) * power;
        if (jigRotation)
            transform.localRotation = baseRotation * Quaternion.Euler(rotationJigAmount * Mathf.Sin(rotationTime) * power);
        if (jigScale)
            transform.localScale = baseScale + scaleJigAmount * Mathf.Sin(scaleTime) * power;
    }
}
