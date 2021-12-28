using TMPro;
using UnityEngine;

/// <summary>
/// A class that handle the title and body of a TimeDisplayItem.
/// </summary>
public class TimeDisplayItem : MonoBehaviour
{
    [Tooltip("A reference to the TextMeshProUGUI to display the time.")]
    [SerializeField]
    protected TextMeshProUGUI display;

    [Tooltip("A reference to the TextMeshProUGUI to display the title for the time.")]
    [SerializeField]
    protected TextMeshProUGUI title;
    
    /// <summary>
    /// Set the text body of the TimeDisplayItem. 
    /// If the input "text" is null or empty the method will disable the TimeDisplayItem gameobject. Otherwise it enables it.
    /// </summary>
    /// <param name="text">string to display in the body</param>
    public void SetText(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            gameObject.SetActive(false);
            return;
        }
        
        gameObject.SetActive(true);
        display.text = text;
    }

    /// <summary>
    /// Set the text title of the TimeDisplayItem.
    /// </summary>
    /// <param name="text">string to display in the title</param>
    public void SetTitle(string text)
    {
        title.text = text;
    }

}
