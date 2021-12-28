using TMPro;
using UnityEngine;

public class VersionDisplay : MonoBehaviour
{
    public TextMeshProUGUI m_Text;
    
    void Awake()
    {
        m_Text.text = $"Version: {Application.version}\n";
    }

    
}
