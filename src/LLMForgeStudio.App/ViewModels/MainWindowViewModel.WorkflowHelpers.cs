namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private string[] GetTrainingMissingSteps()
    {
        var missing = new List<string>();
        if (!_wizardDatasetImported) missing.Add("import dataset");
        if (!IsTokenizerReady) missing.Add("train tokenizer");
        return missing.ToArray();
    }

    private void ResetPreparationState()
    {
        _lastTokenization = null;
        _wizardTokenizerTrained = false;
        _wizardPreviewBuilt = false;
        VocabularyPreview.Clear();
        TokenizerStatusText = "Tokenizer not trained yet.";
        BatchPreviewStatusText = "x/y preview not built yet.";
        OnPropertyChanged(nameof(IsTokenizerReady));
        OnPropertyChanged(nameof(IsBatchPreviewReady));
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
    }
}
