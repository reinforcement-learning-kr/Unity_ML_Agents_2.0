using UnityEngine;
using UnityEngine.UI;

public class FeedbackFlashHUD : MonoBehaviour
{
    [Header("References")]
  
    [Tooltip("CanvasGroup to fade the critical time vignette")]
    public CanvasGroup vignetteCanvasGroup;

    [Header("Critical time")]
    [Tooltip("Max alpha of the critical vignette")]
    public float criticalTimeVignetteMaxAlpha = .8f;
    [Tooltip("Frequency at which the vignette will pulse when at critical health")]
    public float pulsatingVignetteFrequency = 4f;
    [Tooltip("Show the critical time vignette when the remaining time reaches this point")]
    public float showCriticalTimeVignetteWhen = 5f;
    [Tooltip("Audio clip for the critical time")]
    public AudioClip warningAudioClip;

    bool m_FlashActive;

    private TimeManager m_timeManager;
    GameFlowManager m_GameFlowManager;
    AudioSource m_audioSource;
    bool warningSoundPlayed = false;

    
    void Start()
    {
        // Subscribe to player damage events
        
        m_GameFlowManager = FindObjectOfType<GameFlowManager>();
        DebugUtility.HandleErrorIfNullFindObject<GameFlowManager, FeedbackFlashHUD>(m_GameFlowManager, this);

        m_timeManager = FindObjectOfType<TimeManager>();
        DebugUtility.HandleErrorIfNullFindObject<TimeManager, FeedbackFlashHUD>(m_timeManager, this);

        m_audioSource = GetComponent<AudioSource>();
        DebugUtility.HandleErrorIfNullFindObject<AudioSource, FeedbackFlashHUD>(m_audioSource, this);
    }

    private void Update()
    {
        if (!m_timeManager.IsFinite) return;
        
        
        if (m_timeManager.TimeRemaining < showCriticalTimeVignetteWhen)
        {
            EnableFlash(true);
            float vignetteAlpha = criticalTimeVignetteMaxAlpha;

            if (m_GameFlowManager.gameState == GameState.Lost)
                vignetteCanvasGroup.alpha = vignetteAlpha;
            if (m_GameFlowManager.gameState == GameState.Won)
                vignetteCanvasGroup.alpha = 0;
            else
            {
                vignetteCanvasGroup.alpha = ((Mathf.Sin(Time.time * pulsatingVignetteFrequency) / 2) + 0.5f) * vignetteAlpha;

                if(!warningSoundPlayed && vignetteCanvasGroup.alpha >= 0.5f){
                    m_audioSource.PlayOneShot(warningAudioClip);
                    warningSoundPlayed = true;
                }

                if(vignetteCanvasGroup.alpha < 0.5f){
                    warningSoundPlayed = false;
                }

            }
        }
        else if (m_timeManager.TimeRemaining > showCriticalTimeVignetteWhen)
        {
            EnableFlash(false);
        }
        
    }

    private void EnableFlash(bool set)
    {
        if (m_FlashActive == set) return;
        
        vignetteCanvasGroup.gameObject.SetActive(set);
        m_FlashActive = set;
        
        if (!set)  vignetteCanvasGroup.alpha = 0;
    }
  
}
