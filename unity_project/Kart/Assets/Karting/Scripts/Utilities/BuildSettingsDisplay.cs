using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace KartGame

{
    public class BuildSettingsDisplay : MonoBehaviour
    {
        public TextMesh buildName;

        public BuildSettings buildSettings;
        
        void Awake()
        
        {
            buildName.text = $"{buildSettings.buildName}\n{buildSettings.buildType}\n{buildSettings.shaderType}\n";
            buildName.text += $"Resolution: {Screen.width} {Screen.height}";
            gameObject.SetActive(false);
                

        }
    }
}