using KartGame.KartSystems;
using TMPro;
using UnityEngine;

namespace KartGame.UI
{
    public class InGameInfo : MonoBehaviour
    {
        public TextMeshProUGUI Speed;
        public bool AutoFindKart = true;
        public ArcadeKart KartController;

        void Start()
        {
            if (AutoFindKart)
            {
                ArcadeKart kart = FindObjectOfType<ArcadeKart>();
                KartController = kart;
            }

            if (!KartController)
            {
                gameObject.SetActive(false);
            }
        }

        // Update is called once per frame
        void Update()
        {
            float speed = KartController.Rigidbody.velocity.magnitude;
            Speed.text = string.Format($"{Mathf.FloorToInt(speed * 3.6f)} km/h");
            Speed.text += string.Format($"\n{speed:0.0} m/s");
        }
    }
}