using System;
using UnityEngine;

namespace KartGame.KartSystems
{
    [DefaultExecutionOrder(100)]
    public class KartAnimation : MonoBehaviour
    {
        [Serializable] public class Wheel
        {
            [Tooltip("A reference to the transform of the wheel.")]
            public Transform wheelTransform;
            [Tooltip("A reference to the WheelCollider of the wheel.")]
            public WheelCollider wheelCollider;
            
            Quaternion m_SteerlessLocalRotation;

            public void Setup() => m_SteerlessLocalRotation = wheelTransform.localRotation;

            public void StoreDefaultRotation() => m_SteerlessLocalRotation = wheelTransform.localRotation;
            public void SetToDefaultRotation() => wheelTransform.localRotation = m_SteerlessLocalRotation;
        }

        [Tooltip("What kart do we want to listen to?")]
        public ArcadeKart kartController;

        [Space]
        [Tooltip("The damping for the appearance of steering compared to the input.  The higher the number the less damping.")]
        public float steeringAnimationDamping = 10f;

        [Space]
        [Tooltip("The maximum angle in degrees that the front wheels can be turned away from their default positions, when the Steering input is either 1 or -1.")]
        public float maxSteeringAngle;
        [Tooltip("Information referring to the front left wheel of the kart.")]
        public Wheel frontLeftWheel;
        [Tooltip("Information referring to the front right wheel of the kart.")]
        public Wheel frontRightWheel;
        [Tooltip("Information referring to the rear left wheel of the kart.")]
        public Wheel rearLeftWheel;
        [Tooltip("Information referring to the rear right wheel of the kart.")]
        public Wheel rearRightWheel;


        float m_SmoothedSteeringInput;

        void Start()
        {
            frontLeftWheel.Setup();
            frontRightWheel.Setup();
            rearLeftWheel.Setup();
            rearRightWheel.Setup();
        }

        void FixedUpdate() 
        {
            m_SmoothedSteeringInput = Mathf.MoveTowards(m_SmoothedSteeringInput, kartController.Input.TurnInput, 
                steeringAnimationDamping * Time.deltaTime);

            // Steer front wheels
            float rotationAngle = m_SmoothedSteeringInput * maxSteeringAngle;

            frontLeftWheel.wheelCollider.steerAngle = rotationAngle;
            frontRightWheel.wheelCollider.steerAngle = rotationAngle;

            // Update position and rotation from WheelCollider
            UpdateWheelFromCollider(frontLeftWheel);
            UpdateWheelFromCollider(frontRightWheel);
            UpdateWheelFromCollider(rearLeftWheel);
            UpdateWheelFromCollider(rearRightWheel);
        }

        void LateUpdate()
        {
            // Update position and rotation from WheelCollider
            UpdateWheelFromCollider(frontLeftWheel);
            UpdateWheelFromCollider(frontRightWheel);
            UpdateWheelFromCollider(rearLeftWheel);
            UpdateWheelFromCollider(rearRightWheel);
        }

        void UpdateWheelFromCollider(Wheel wheel)
        {
            wheel.wheelCollider.GetWorldPose(out Vector3 position, out Quaternion rotation);
            wheel.wheelTransform.position = position;
            wheel.wheelTransform.rotation = rotation;
        }
    }
}
