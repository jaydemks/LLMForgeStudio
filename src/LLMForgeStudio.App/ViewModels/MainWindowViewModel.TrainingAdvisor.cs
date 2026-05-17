using Avalonia.Media;

namespace LLMForgeStudio.App.ViewModels;

public sealed partial class MainWindowViewModel
{
    private static readonly IBrush AdvisorAlertBrush = new SolidColorBrush(Color.Parse("#EF4444"));

    private string _trainingBatchAdvisorText = string.Empty;
    private string _trainingMaxStepsAdvisorText = string.Empty;
    private string _trainingLearningRateAdvisorText = string.Empty;
    private string _trainingEvalEveryAdvisorText = string.Empty;
    private string _trainingGradientClippingAdvisorText = string.Empty;
    private string _trainingDedupAdvisorText = string.Empty;
    private string _trainingDedupParagraphsAdvisorText = string.Empty;
    private string _trainingCollapseWhitespaceAdvisorText = string.Empty;
    private string _trainingCurriculumLearningAdvisorText = string.Empty;

    public string TrainingBatchAdvisorText
    {
        get => _trainingBatchAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingBatchAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingBatchAdvisorVisible));
                OnPropertyChanged(nameof(TrainingBatchLabelBrush));
            }
        }
    }

    public string TrainingMaxStepsAdvisorText
    {
        get => _trainingMaxStepsAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingMaxStepsAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingMaxStepsAdvisorVisible));
                OnPropertyChanged(nameof(TrainingMaxStepsLabelBrush));
            }
        }
    }

    public string TrainingLearningRateAdvisorText
    {
        get => _trainingLearningRateAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingLearningRateAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingLearningRateAdvisorVisible));
                OnPropertyChanged(nameof(TrainingLearningRateLabelBrush));
            }
        }
    }

    public string TrainingEvalEveryAdvisorText
    {
        get => _trainingEvalEveryAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingEvalEveryAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingEvalEveryAdvisorVisible));
                OnPropertyChanged(nameof(TrainingEvalEveryLabelBrush));
            }
        }
    }

    public bool TrainingBatchAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingBatchAdvisorText);
    public bool TrainingMaxStepsAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingMaxStepsAdvisorText);
    public bool TrainingLearningRateAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingLearningRateAdvisorText);
    public bool TrainingEvalEveryAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingEvalEveryAdvisorText);
    public bool TrainingGradientClippingAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingGradientClippingAdvisorText);
    public bool TrainingDedupAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingDedupAdvisorText);
    public bool TrainingDedupParagraphsAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingDedupParagraphsAdvisorText);
    public bool TrainingCollapseWhitespaceAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingCollapseWhitespaceAdvisorText);
    public bool TrainingCurriculumLearningAdvisorVisible => !string.IsNullOrWhiteSpace(TrainingCurriculumLearningAdvisorText);

    public IBrush? TrainingBatchLabelBrush => TrainingBatchAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingMaxStepsLabelBrush => TrainingMaxStepsAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingLearningRateLabelBrush => TrainingLearningRateAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingEvalEveryLabelBrush => TrainingEvalEveryAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingGradientClippingLabelBrush => TrainingGradientClippingAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingDedupLabelBrush => TrainingDedupAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingDedupParagraphsLabelBrush => TrainingDedupParagraphsAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingCollapseWhitespaceLabelBrush => TrainingCollapseWhitespaceAdvisorVisible ? AdvisorAlertBrush : null;
    public IBrush? TrainingCurriculumLearningLabelBrush => TrainingCurriculumLearningAdvisorVisible ? AdvisorAlertBrush : null;

    public string TrainingGradientClippingAdvisorText
    {
        get => _trainingGradientClippingAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingGradientClippingAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingGradientClippingAdvisorVisible));
                OnPropertyChanged(nameof(TrainingGradientClippingLabelBrush));
            }
        }
    }

    public string TrainingDedupAdvisorText
    {
        get => _trainingDedupAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingDedupAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingDedupAdvisorVisible));
                OnPropertyChanged(nameof(TrainingDedupLabelBrush));
            }
        }
    }

    public string TrainingDedupParagraphsAdvisorText
    {
        get => _trainingDedupParagraphsAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingDedupParagraphsAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingDedupParagraphsAdvisorVisible));
                OnPropertyChanged(nameof(TrainingDedupParagraphsLabelBrush));
            }
        }
    }

    public string TrainingCollapseWhitespaceAdvisorText
    {
        get => _trainingCollapseWhitespaceAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingCollapseWhitespaceAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingCollapseWhitespaceAdvisorVisible));
                OnPropertyChanged(nameof(TrainingCollapseWhitespaceLabelBrush));
            }
        }
    }

    public string TrainingCurriculumLearningAdvisorText
    {
        get => _trainingCurriculumLearningAdvisorText;
        private set
        {
            if (SetProperty(ref _trainingCurriculumLearningAdvisorText, value))
            {
                OnPropertyChanged(nameof(TrainingCurriculumLearningAdvisorVisible));
                OnPropertyChanged(nameof(TrainingCurriculumLearningLabelBrush));
            }
        }
    }

    private void ComputeTrainingAdvisorFromLogs()
    {
        if (TrainingLogs.Count == 0) return;
        var last = TrainingLogs.Last();
        var bestVal = TrainingLogs.Min(x => x.ValLoss);
        var gap = last.ValLoss - last.TrainLoss;

        ClearAllTrainingAdvisor();

        // Overfitting signal.
        if (gap >= 2.0 && last.ValLoss >= 4.0)
        {
            var suggestedSteps = Math.Max(300, (int)Math.Round(TrainingConfig.MaxSteps * 0.8));
            var suggestedLr = Math.Max(1e-6, TrainingConfig.LearningRate * 0.7);
            var suggestedEval = Math.Max(10, TrainingConfig.EvalEvery / 2);
            TrainingMaxStepsAdvisorText = IsEnglish
                ? $"↓ try ~{suggestedSteps:N0} (reduce overfit)"
                : $"↓ prova ~{suggestedSteps:N0} (riduci overfit)";
            TrainingLearningRateAdvisorText = IsEnglish
                ? $"↓ try {suggestedLr:0.######}"
                : $"↓ prova {suggestedLr:0.######}";
            TrainingEvalEveryAdvisorText = IsEnglish
                ? $"↓ try {suggestedEval:N0} (more frequent eval)"
                : $"↓ prova {suggestedEval:N0} (eval più frequente)";
            if (!TrainingConfig.EnableGradientClipping)
                TrainingGradientClippingAdvisorText = IsEnglish ? "↑ enable gradient clipping" : "↑ attiva gradient clipping";
            if (!TrainingConfig.EnableDeduplication)
                TrainingDedupAdvisorText = IsEnglish ? "↑ enable dataset dedup" : "↑ attiva dedup dataset";
            if (!TrainingConfig.RemoveDuplicateParagraphs)
                TrainingDedupParagraphsAdvisorText = IsEnglish ? "↑ enable paragraph dedup" : "↑ attiva dedup paragrafi";
            if (!TrainingConfig.CollapseWhitespace)
                TrainingCollapseWhitespaceAdvisorText = IsEnglish ? "↑ enable whitespace collapse" : "↑ attiva collapse whitespace";
            if (!TrainingConfig.CurriculumLearning)
                TrainingCurriculumLearningAdvisorText = IsEnglish ? "↑ enable curriculum learning" : "↑ attiva curriculum learning";
            return;
        }

        // Underfitting / weak learning signal.
        if (last.TrainLoss >= 4.0 && last.ValLoss >= 4.5)
        {
            var suggestedSteps = Math.Min(1_000_000, (int)Math.Round(TrainingConfig.MaxSteps * 1.25));
            var suggestedBatch = Math.Max(1, TrainingConfig.BatchSize / 2);
            TrainingMaxStepsAdvisorText = IsEnglish
                ? $"↑ try ~{suggestedSteps:N0} (more learning)"
                : $"↑ prova ~{suggestedSteps:N0} (più apprendimento)";
            if (suggestedBatch != TrainingConfig.BatchSize)
            {
                TrainingBatchAdvisorText = IsEnglish
                    ? $"↓ try {suggestedBatch:N0} (higher update noise)"
                    : $"↓ prova {suggestedBatch:N0} (più rumore utile)";
            }
            if (TrainingConfig.CurriculumLearning && TrainingConfig.CurriculumWarmupRatio > 0.65)
                TrainingCurriculumLearningAdvisorText = IsEnglish ? "↓ lower curriculum warmup bias" : "↓ riduci bias warmup curriculum";
            return;
        }

        // Best val significantly better than last val -> likely late over-training.
        if (bestVal + 0.8 < last.ValLoss)
        {
            var suggestedSteps = Math.Max(300, (int)Math.Round(TrainingConfig.MaxSteps * 0.85));
            TrainingMaxStepsAdvisorText = IsEnglish
                ? $"↓ stop earlier, try ~{suggestedSteps:N0}"
                : $"↓ fermati prima, prova ~{suggestedSteps:N0}";
        }
    }

    private void ClearTrainingAdvisorForField(string fieldName)
    {
        switch (fieldName)
        {
            case nameof(TrainingBatchSize):
                TrainingBatchAdvisorText = string.Empty;
                break;
            case nameof(TrainingMaxSteps):
                TrainingMaxStepsAdvisorText = string.Empty;
                break;
            case nameof(TrainingLearningRate):
                TrainingLearningRateAdvisorText = string.Empty;
                break;
            case nameof(TrainingEvalEvery):
                TrainingEvalEveryAdvisorText = string.Empty;
                break;
            case nameof(TrainingGradientClipping):
                TrainingGradientClippingAdvisorText = string.Empty;
                break;
            case nameof(TrainingDedup):
                TrainingDedupAdvisorText = string.Empty;
                break;
            case nameof(TrainingDedupParagraphs):
                TrainingDedupParagraphsAdvisorText = string.Empty;
                break;
            case nameof(TrainingCollapseWhitespace):
                TrainingCollapseWhitespaceAdvisorText = string.Empty;
                break;
            case nameof(TrainingCurriculumLearning):
                TrainingCurriculumLearningAdvisorText = string.Empty;
                break;
        }
    }

    private void ClearAllTrainingAdvisor()
    {
        TrainingBatchAdvisorText = string.Empty;
        TrainingMaxStepsAdvisorText = string.Empty;
        TrainingLearningRateAdvisorText = string.Empty;
        TrainingEvalEveryAdvisorText = string.Empty;
        TrainingGradientClippingAdvisorText = string.Empty;
        TrainingDedupAdvisorText = string.Empty;
        TrainingDedupParagraphsAdvisorText = string.Empty;
        TrainingCollapseWhitespaceAdvisorText = string.Empty;
        TrainingCurriculumLearningAdvisorText = string.Empty;
    }
}
