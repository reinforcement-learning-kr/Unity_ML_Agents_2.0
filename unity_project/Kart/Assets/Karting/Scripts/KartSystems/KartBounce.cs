using UnityEngine;

namespace KartGame.KartSystems
{

    /// <summary>
    /// Allows custom collision detection and bounce effects when the kart collides to the collision layer.
    /// </summary>
    [RequireComponent(typeof(ArcadeKart))]
    public class KartBounce : MonoBehaviour
    {
        /// <summary>
        /// Represents a single frame where the bounce was actually triggered. Use this as a flag 
        /// to detect when bouncing was invoked. This flag is managed by the VehicleBounce.
        /// </summary>
        public bool BounceFlag { get; private set; }

        [Tooltip("How much impulse should be applied to the kart when it collides?")]
        public float BounceFactor = 10f;
        [Tooltip("How fast should the kart reorient itself when colliding? The higher the value the more snappy it looks")]
        public float RotationSpeed = 3f;
        [Tooltip("What layer should this GameObject collider with?")]
        public LayerMask CollisionLayer;
        [Tooltip("How far ahead should the vehicle detect a bounce? This should be a positive value!")]
        public float RayDistance = 1f;
        [Tooltip("When can the kart bounce again if it bounced once?")]
        public float PauseTime = 0.5f;
        [Tooltip("Do we need to adjust the y height of the ray's origin to compensate for larger vehicles?")]
        public float HeightOffset;
        [Tooltip("How many raycasts do we intend to shoot out to detect the bounce and at what angles are we doing so? " +
            "Enable DrawGizmos to help you debug what the ray casts look like when selecting angles.")]
        public float[] Angles;

        [Tooltip("Should gizmos be drawn for debugging purposes? This is helpful for checking the rays.")]
        public bool DrawGizmos;
        [Tooltip("What audio clip should play when the kart collides with a wall")]
        public AudioClip BounceSound;

        ArcadeKart kart;
        float resumeTime;
        bool hasCollided;
        Vector3 reflectionVector;

        void Start()
        {
            kart = GetComponent<ArcadeKart>();
        }

        void Update()
        {
            // Reset the trigger flag
            if (BounceFlag)
            {
                BounceFlag = false;
            }
            Vector3 origin = transform.position;
            origin.y += HeightOffset;

            for (int i = 0; i < Angles.Length; i++)
            {
                Vector3 direction = GetDirectionFromAngle(Angles[i], Vector3.up, transform.forward);

                if (Physics.Raycast(origin, direction, out RaycastHit hit, RayDistance, CollisionLayer) && Time.time > resumeTime && !hasCollided && kart.LocalSpeed() > 0)
                {
                    // If the hit normal is pointing up, then we don't want to bounce
                    if (Vector3.Dot(hit.normal, Vector3.up) > 0.2f) 
                    { 
                        return;
                    }

                    // Calculate the incident vector of the kart colliding into whatever object
                    Vector3 incidentVector =  hit.point - origin;

                    // Calculate the reflection vector using the incident vector of the collision
                    Vector3 hitNormal = hit.normal.normalized;
                    reflectionVector = incidentVector - 2 * Vector3.Dot(incidentVector, hitNormal) * hitNormal;
                    reflectionVector.y = 0;

                    kart.Rigidbody.velocity /= 2;
                    // Apply the bounce impulse with the reflectionVector
                    kart.Rigidbody.AddForce(reflectionVector.normalized * BounceFactor, ForceMode.Impulse);

                    // Mark that the vehicle has collided and the reset time.
                    kart.SetCanMove(false);
                    BounceFlag = hasCollided = true;
                    resumeTime = Time.time + PauseTime;

                    if (BounceSound)
                    {
                        AudioUtility.CreateSFX(BounceSound, transform.position, AudioUtility.AudioGroups.Collision, 0f);
                    }
                    return;
                }
            }

            if (Time.time < resumeTime)
            {
                Vector3 targetPos         = origin + reflectionVector;
                Vector3 direction         = targetPos - origin;
                Quaternion targetRotation = Quaternion.LookRotation(direction);
                kart.transform.rotation   = Quaternion.Slerp(kart.transform.rotation, targetRotation, RotationSpeed * Time.deltaTime);
            }
        }

        void LateUpdate()
        {
            if (Time.time > resumeTime && hasCollided) 
            {
                kart.SetCanMove(true);
                hasCollided = false;
            }
        }

        void OnDrawGizmos()
        {
            if (DrawGizmos)
            {
                Vector3 origin = transform.position;
                origin.y += HeightOffset;

                Gizmos.color = Color.cyan;
                Gizmos.DrawLine(origin, origin + transform.forward);
                Gizmos.color = Color.red;
                for (int i = 0; i < Angles.Length; i++)
                {
                    var direction = GetDirectionFromAngle(Angles[i], Vector3.up, transform.forward);
                    Gizmos.DrawLine(origin, origin + (direction.normalized * RayDistance));
                }
            }
        }

        Vector3 GetDirectionFromAngle(float degrees, Vector3 axis, Vector3 zerothDirection)
        {
            Quaternion rotation = Quaternion.AngleAxis(degrees, axis);
            return (rotation * zerothDirection);
        }
    }
}
