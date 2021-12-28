using UnityEngine;

namespace KartGame
{
    public class SimpleShaker : MonoBehaviour
    {
        Vector3 basePos;
        Quaternion baseRot;
        public float shakeAmount = .1f;
        public float rotationShakeAmount = .1f;
        public float frequency = 10;

        float seed1;
        float seed2;
        float seed3;

        // Start is called before the first frame update
        void Start()
        {
            basePos = transform.localPosition;
            baseRot = transform.localRotation;
            seed1 = Random.Range(0, 999);
            seed2 = Random.Range(0, 999);
            seed3 = Random.Range(0, 999);
        }

        // Update is called once per frame
        void Update()
        {
            transform.localPosition = basePos + shakeAmount * new Vector3(
                Mathf.PerlinNoise(Time.time * frequency, seed1)-.5f,
                Mathf.PerlinNoise(Time.time * frequency, seed2)-.5f,
                Mathf.PerlinNoise(Time.time * frequency, seed3)-.5f);

            var rotationNoise = new Vector3(
                Mathf.PerlinNoise(Time.time * frequency, seed3) - .5f,
                Mathf.PerlinNoise(Time.time * frequency, seed2) - .5f,
                Mathf.PerlinNoise(Time.time * frequency, seed1) - .5f);

            transform.localRotation = Quaternion.Euler( rotationNoise * rotationShakeAmount) * baseRot;
        }
    }
}
