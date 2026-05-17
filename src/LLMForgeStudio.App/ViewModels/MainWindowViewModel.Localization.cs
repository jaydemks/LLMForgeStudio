using Avalonia;
using Avalonia.Styling;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private void SetLanguage(bool english)
    {
        IsEnglish = english;
        UpdateSectionsForLanguage();
        RecomputeDatasetRecommendations(null);
    }

    private void ToggleTheme()
    {
        ApplyTheme(!IsLightTheme);
    }

    private void ApplyTheme(bool light)
    {
        IsLightTheme = light;
        if (Application.Current is not null)
            Application.Current.RequestedThemeVariant = IsLightTheme ? ThemeVariant.Light : ThemeVariant.Dark;
        OnPropertyChanged(nameof(HeaderBackground));
        OnPropertyChanged(nameof(SidebarBackground));
        OnPropertyChanged(nameof(DividerBrush));
        OnPropertyChanged(nameof(ModalBackground));
        OnPropertyChanged(nameof(ModalBorderBrush));
    }

    private void RefreshLocalizedTexts()
    {
        OnPropertyChanged(nameof(SettingsTitle));
        OnPropertyChanged(nameof(AboutTitle));
        OnPropertyChanged(nameof(AboutText));
        OnPropertyChanged(nameof(LanguageLabel));
        OnPropertyChanged(nameof(ThemeLabel));
        OnPropertyChanged(nameof(LightLabel));
        OnPropertyChanged(nameof(DarkLabel));
        OnPropertyChanged(nameof(WizardToggleText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
        OnPropertyChanged(nameof(SectionWorkflowText));
        OnPropertyChanged(nameof(ButtonLoadProject));
        OnPropertyChanged(nameof(ButtonSaveProject));
        OnPropertyChanged(nameof(ButtonLogs));
        OnPropertyChanged(nameof(ButtonGuide));
        OnPropertyChanged(nameof(ButtonTrainTokenizer));
        OnPropertyChanged(nameof(AdvancedTrainingHeaderText));
        OnPropertyChanged(nameof(WelcomeTitle));
        OnPropertyChanged(nameof(HeaderSubtitleText));
        OnPropertyChanged(nameof(GuideTitleText));
        OnPropertyChanged(nameof(GuideIntroText));
        OnPropertyChanged(nameof(GuideThanksText));
        OnPropertyChanged(nameof(GuideDatasetText));
        OnPropertyChanged(nameof(GuideTokenizationText));
        OnPropertyChanged(nameof(GuideTokenizationKeyText));
        OnPropertyChanged(nameof(GuideModelText));
        OnPropertyChanged(nameof(GuideModelKeyText));
        OnPropertyChanged(nameof(GuideTrainingText));
        OnPropertyChanged(nameof(GuideTrainingKeyText));
        OnPropertyChanged(nameof(GuideGenerationText));
        OnPropertyChanged(nameof(GuideGenerationKeyText));
        OnPropertyChanged(nameof(ModelContextWindowText));
        OnPropertyChanged(nameof(GenerationOutputWindowText));
        OnPropertyChanged(nameof(GuideOrderText));
        OnPropertyChanged(nameof(GuideSourceText));
        OnPropertyChanged(nameof(HardwareAutoDetectText));
        OnPropertyChanged(nameof(TokenizationStepGuide));
        OnPropertyChanged(nameof(TrainingStepGuide));
        OnPropertyChanged(nameof(TrainingWizardMiniText));
        OnPropertyChanged(nameof(TrainingPreflightText));
        OnPropertyChanged(nameof(DatasetHelpText));
        OnPropertyChanged(nameof(DatasetCopyPathText));
        OnPropertyChanged(nameof(DatasetPathTooltipText));
        OnPropertyChanged(nameof(DatasetStatsCharactersLabel));
        OnPropertyChanged(nameof(DatasetStatsLinesLabel));
        OnPropertyChanged(nameof(DatasetStatsWordsLabel));
        OnPropertyChanged(nameof(DatasetStatsUniqueCharsLabel));
        OnPropertyChanged(nameof(DatasetPreviewLabelText));
        OnPropertyChanged(nameof(DatasetPreviewHintText));
        OnPropertyChanged(nameof(DatasetRecommendationsTitle));
        OnPropertyChanged(nameof(DatasetApplyRecommendationsText));
        OnPropertyChanged(nameof(DatasetRecommendedTokenizerLine));
        OnPropertyChanged(nameof(DatasetRecommendedTrainingProfileLine));
        OnPropertyChanged(nameof(DatasetRecommendationsText));
        OnPropertyChanged(nameof(RunDirectoryHelpText));
        OnPropertyChanged(nameof(GatherTitleText));
        OnPropertyChanged(nameof(GatherIntroText));
        OnPropertyChanged(nameof(GatherSourceLabel));
        OnPropertyChanged(nameof(GatherSourceTooltipText));
        OnPropertyChanged(nameof(GatherWorkspaceLabel));
        OnPropertyChanged(nameof(GatherBrowseText));
        OnPropertyChanged(nameof(GatherWorkspaceButtonText));
        OnPropertyChanged(nameof(GatherLicenseGateTitle));
        OnPropertyChanged(nameof(GatherCheckLicenseButtonText));
        OnPropertyChanged(nameof(GatherStep1CheckLicenseText));
        OnPropertyChanged(nameof(GatherLicenseAckText));
        OnPropertyChanged(nameof(GatherFetchButtonText));
        OnPropertyChanged(nameof(GatherStep2FetchText));
        OnPropertyChanged(nameof(GatherConvertButtonText));
        OnPropertyChanged(nameof(GatherStep3ConvertText));
        OnPropertyChanged(nameof(GatherValidateButtonText));
        OnPropertyChanged(nameof(GatherStep5ValidateText));
        OnPropertyChanged(nameof(GatherApplyButtonText));
        OnPropertyChanged(nameof(GatherStep6ApplyText));
        OnPropertyChanged(nameof(GatherMergeButtonText));
        OnPropertyChanged(nameof(GatherStep4MergeText));
        OnPropertyChanged(nameof(GatherClearButtonText));
        OnPropertyChanged(nameof(GatherCancelButtonText));
        OnPropertyChanged(nameof(GatherRemoveSourceButtonText));
        OnPropertyChanged(nameof(GatherHandoffScratchText));
        OnPropertyChanged(nameof(GatherStep7HandoffScratchText));
        OnPropertyChanged(nameof(GatherHandoffFtText));
        OnPropertyChanged(nameof(GatherStep7HandoffFtText));
        OnPropertyChanged(nameof(GatherRecTokenizerText));
        OnPropertyChanged(nameof(GatherRecProfileText));
        OnPropertyChanged(nameof(GatherMergeOptionsTitle));
        OnPropertyChanged(nameof(GatherDedupPolicyLabel));
        OnPropertyChanged(nameof(GatherSourcesTableTitle));
        OnPropertyChanged(nameof(GatherFlowHintText));
        OnPropertyChanged(nameof(GatherSourcesCountText));
        OnPropertyChanged(nameof(GatherParquetHintText));
        OnPropertyChanged(nameof(GatherNeedsParquetConversion));
        OnPropertyChanged(nameof(GatherRecommendedTokenizerLine));
        OnPropertyChanged(nameof(GatherRecommendedTrainingProfileLine));
        OnPropertyChanged(nameof(LogsOverlayTitle));
        OnPropertyChanged(nameof(LogsOverlaySizeLabel));
        OnPropertyChanged(nameof(LogsOverlayWidthLabel));
        OnPropertyChanged(nameof(LogsOverlayHeightLabel));
        OnPropertyChanged(nameof(LogsOverlayClearButton));
        OnPropertyChanged(nameof(LogsOverlayCloseButton));
    }

    private void UpdateSectionsForLanguage()
    {
        var previous = SelectedSection;
        var items = IsEnglish
            ? new[] { "Hardware", "Gather Dataset", "Dataset", "Tokenization", "Model", "Training", "Generation", "Fine-Tuning (Ollama, Experimental)" }
            : new[] { "Hardware", "Raccolta Dataset", "Dataset", "Tokenization", "Model", "Training", "Generazione", "Fine-Tuning (Ollama, Experimental)" };
        Sections.Clear();
        foreach (var it in items) Sections.Add(it);

        SelectedSection = previous switch
        {
            "Gather Dataset" when !IsEnglish => "Raccolta Dataset",
            "Raccolta Dataset" when IsEnglish => "Gather Dataset",
            "Generation" when !IsEnglish => "Generazione",
            "Generazione" when IsEnglish => "Generation",
            "Guide" when !IsEnglish => "Guida",
            "Guida" when IsEnglish => "Guide",
            _ => Sections.Contains(previous) ? previous : Sections.First()
        };
    }
}
