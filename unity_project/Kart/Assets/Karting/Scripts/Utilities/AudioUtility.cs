using System.Collections;
using UnityEngine;
using UnityEngine.Audio;

public class AudioUtility
{
    static AudioManager m_AudioManager;

    public enum AudioGroups
    {
        Collision,
        Pickup,
        HUDVictory,
        HUDObjective
    }

    public static void CreateSFX(AudioClip clip, Vector3 position, AudioGroups audioGroup, float spatialBlend, float rolloffDistanceMin = 1f)
    {
        GameObject impactSFXInstance = new GameObject("SFX_" + clip.name);
        impactSFXInstance.transform.position = position;
        AudioSource source = impactSFXInstance.AddComponent<AudioSource>();
        source.clip = clip;
        source.spatialBlend = spatialBlend;
        source.minDistance = rolloffDistanceMin;
        source.Play();

        source.outputAudioMixerGroup = GetAudioGroup(audioGroup);

        m_AudioManager.EnsureSFXDestruction(source);
    }

    public static AudioMixerGroup GetAudioGroup(AudioGroups group)
    {
        if (m_AudioManager == null)
            m_AudioManager = GameObject.FindObjectOfType<AudioManager>();

        var groups = m_AudioManager.audioMixer.FindMatchingGroups(group.ToString());

        if (groups.Length > 0)
            return groups[0];

        Debug.LogWarning("Didn't find audio group for " + group.ToString());
        return null;
    }

    public static void SetMasterVolume(float value)
    {
        if (m_AudioManager == null)
            m_AudioManager = GameObject.FindObjectOfType<AudioManager>();

        if (value <= 0)
            value = 0.001f;
        float valueInDB = Mathf.Log10(value) * 20;

        m_AudioManager.audioMixer.SetFloat("MasterVolume", valueInDB);
    }

    public static float GetMasterVolume()
    {
        if (m_AudioManager == null)
            m_AudioManager = GameObject.FindObjectOfType<AudioManager>();

        m_AudioManager.audioMixer.GetFloat("MasterVolume", out var valueInDB);
        return Mathf.Pow(10f, valueInDB / 20.0f);
    }
}
