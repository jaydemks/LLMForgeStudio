using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Windows.Input;
using Avalonia;
using Avalonia.Styling;
using Avalonia.Threading;
using LLMForgeStudio.App.Core.Backend;
using LLMForgeStudio.App.Core.Alignment;
using LLMForgeStudio.App.Core.Dataset;
using LLMForgeStudio.App.Core.Diagnostics;
using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Cluster;
using LLMForgeStudio.App.Core.Guidance;
using LLMForgeStudio.App.Core.Hardware;
using LLMForgeStudio.App.Core.Project;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.ViewModels;

public sealed class MainWindowViewModel : ObservableObject
{
    private enum BackendFlavor
    {
        NvidiaCuda,
        AmdDirectMl,
        Cpu
    }

    private string _datasetText = "The morning sun rises over a small local model.\nThis is only sample text. Replace it with your dataset.";
    private string _datasetPath = string.Empty;
    private string _selectedSection = "Hardware";
    private TokenizerKind _selectedTokenizerKind = TokenizerKind.Character;
    private TokenizationResult? _lastTokenization;
    private string _log = "Ready. Load a dataset or use the sample text, then train a tokenizer.";
    private string _pythonPath = "python";
    private string _runDirectory = Path.Combine(Environment.CurrentDirectory, "runs", "default");
    private string _checkpointPath = string.Empty;
    private string _generationPrompt = "The morning";
    private string _generatedText = string.Empty;
    private string _generationStatusText = "Generation idle.";
    private bool _isGenerating;
    private bool _isTraining;
    private Process? _trainingProcess;
    private bool _isBackendBootstrapping;
    private bool _backendUserInitiatedStartup;
    private bool _showStartupOverlay = true;
    private int _backendSetupProgress;
    private string _backendSetupStage = "Idle";
    private bool _wizardEnabled = true;
    private bool _showSettingsOverlay;
    private bool _showAboutOverlay;
    private bool _isEnglish = true;
    private bool _isLightTheme;
    private double _cpuUsage;
    private double _gpuUsage;
    private double _ramUsage;
    private double _diskUsage;
    private readonly DispatcherTimer _perfTimer;
    private bool _isPerfSampling;
    private bool _wizardSetupDone;
    private bool _wizardDatasetImported;
    private bool _wizardTokenizerTrained;
    private bool _wizardPreviewBuilt;
    private bool _wizardTrainingStarted;
    private bool _wizardCheckpointSet;
    private string _tokenizerStatusText = "Tokenizer not trained yet.";
    private string _batchPreviewStatusText = "x/y preview not built yet.";
    private string _trainingStatusText = "Training idle.";
    private string _rlhfDraftPrompt = string.Empty;
    private string _rlhfDraftChosen = string.Empty;
    private string _rlhfDraftRejected = string.Empty;
    private DateTimeOffset? _trainingStartedAtUtc;
    private bool _showAdvancedTraining;
    private string _selectedTrainingProfile = "Custom";
    private EvalSummarySnapshot? _lastEvalSummary;
    private readonly UiDebugLogger _uiDebugLogger = new();

    public ObservableCollection<string> Sections { get; } = new(new[] { "Hardware", "Dataset", "Tokenization", "Model", "Training", "Generation", "Guide" });
    public ObservableCollection<TokenizerKind> TokenizerKinds { get; } = new(Enum.GetValues<TokenizerKind>());
    public ObservableCollection<VocabularyItem> VocabularyPreview { get; } = new();
    public ObservableCollection<string> Warnings { get; } = new();
    public ObservableCollection<TrainingLogEntry> TrainingLogs { get; } = new();
    public ObservableCollection<string> HardwareGpus { get; } = new();
    public ObservableCollection<string> OptimizerOptions { get; } = new(new[] { "adamw", "lion", "adafactor" });
    public ObservableCollection<string> SchedulerOptions { get; } = new(new[] { "none", "cosine", "linear" });
    public ObservableCollection<string> PrecisionOptions { get; } = new(new[] { "fp16", "bf16" });
    public ObservableCollection<string> MultiGpuStrategyOptions { get; } = new(new[] { "none", "ddp", "fsdp" });
    public ObservableCollection<string> QuantizationProfileOptions { get; } = new(new[] { "dynamic-int8", "ptq-int8", "ptq-int4" });
    public ObservableCollection<string> AlignmentOptions { get; } = new(new[] { "none", "sft", "dpo", "rlhf" });
    public ObservableCollection<string> RlhfFeedbackOptions { get; } = new(new[] { "inline", "jsonl-human-feedback", "external-import" });
    public ObservableCollection<string> SafetyPolicyOptions { get; } = new(new[] { "standard", "strict", "research" });
    public ObservableCollection<string> EvalSuiteOptions { get; } = new(new[] { "basic", "quick-5", "standard-10", "full-20" });
    public ObservableCollection<string> TrainingProfileOptions { get; } = new(new[] { "Custom", "Tiny", "Balanced", "Serious", "Research", "Cluster" });
    public ObservableCollection<string> ClusterProfileOptions { get; } = new(ClusterProfileManager.BuiltIns.Select(x => x.Name));
    public ObservableCollection<RlhfFeedbackRecord> RlhfCollectedFeedback { get; } = new();

    public TextCleanerConfig Cleaner { get; } = new();
    public TokenizerConfig TokenizerConfig { get; } = new();
    public ModelConfig ModelConfig { get; } = new();
    public TrainingConfig TrainingConfig { get; } = new();
    public SamplingConfig SamplingConfig { get; } = new();

    public ICommand TrainTokenizerCommand { get; }
    public ICommand BuildBatchPreviewCommand { get; }
    public ICommand DryRunTrainingCommand { get; }
    public ICommand GeneratePreviewCommand { get; }
    public ICommand StartBackendTrainingCommand { get; }
    public ICommand CancelBackendTrainingCommand { get; }
    public ICommand GenerateFromCheckpointCommand { get; }
    public ICommand SetupBackendCommand { get; }
    public ICommand DismissStartupOverlayCommand { get; }
    public ICommand ToggleSettingsCommand { get; }
    public ICommand ToggleAboutCommand { get; }
    public ICommand CloseAboutCommand { get; }
    public ICommand ApplyLanguageItCommand { get; }
    public ICommand ApplyLanguageEnCommand { get; }
    public ICommand ToggleThemeCommand { get; }
    public ICommand SetLightThemeCommand { get; }
    public ICommand SetDarkThemeCommand { get; }
    public ICommand WizardGoCommand { get; }
    public ICommand DisableWizardCommand { get; }
    public ICommand AddRlhfFeedbackCommand { get; }
    public ICommand ClearRlhfFeedbackCommand { get; }
    public ICommand OpenGuideCommand { get; }

    public MainWindowViewModel()
    {
        TrainTokenizerCommand = new RelayCommand(TrainTokenizer);
        BuildBatchPreviewCommand = new RelayCommand(BuildBatchPreview);
        DryRunTrainingCommand = new RelayCommand(RunDryTraining);
        GeneratePreviewCommand = new RelayCommand(GeneratePreview);
        StartBackendTrainingCommand = new RelayCommand(() => _ = StartBackendTrainingAsync());
        CancelBackendTrainingCommand = new RelayCommand(CancelBackendTraining);
        GenerateFromCheckpointCommand = new RelayCommand(() => _ = GenerateFromCheckpointAsync());
        SetupBackendCommand = new RelayCommand(() => _ = SetupBackendAsync());
        DismissStartupOverlayCommand = new RelayCommand(() => ShowStartupOverlay = false);
        ToggleSettingsCommand = new RelayCommand(() => ShowSettingsOverlay = !ShowSettingsOverlay);
        ToggleAboutCommand = new RelayCommand(() => ShowAboutOverlay = true);
        CloseAboutCommand = new RelayCommand(() => ShowAboutOverlay = false);
        ApplyLanguageItCommand = new RelayCommand(() => SetLanguage(false));
        ApplyLanguageEnCommand = new RelayCommand(() => SetLanguage(true));
        ToggleThemeCommand = new RelayCommand(ToggleTheme);
        SetLightThemeCommand = new RelayCommand(() => ApplyTheme(true));
        SetDarkThemeCommand = new RelayCommand(() => ApplyTheme(false));
        WizardGoCommand = new RelayCommand(GoToWizardTarget);
        DisableWizardCommand = new RelayCommand(() => WizardEnabled = false);
        AddRlhfFeedbackCommand = new RelayCommand(AddRlhfFeedback);
        ClearRlhfFeedbackCommand = new RelayCommand(ClearRlhfFeedback);
        OpenGuideCommand = new RelayCommand(() => SelectedSection = "Guide");

        // Ensure initial tokenizer UI fields match the default selected tokenizer.
        ApplyRecommendedTokenizerSettings(_selectedTokenizerKind);

        RefreshAll();
        _ = RefreshHardwareAsync();
        _ = BootstrapBackendOnStartupAsync();

        _perfTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(3) };
        _perfTimer.Tick += async (_, _) => await RefreshPerformanceAsync();
        _perfTimer.Start();
    }

    public string SelectedSection
    {
        get => _selectedSection;
        set
        {
            if (!SetProperty(ref _selectedSection, value)) return;
            OnPropertyChanged(nameof(IsDatasetSection));
            OnPropertyChanged(nameof(IsHardwareSection));
            OnPropertyChanged(nameof(IsTokenizationSection));
            OnPropertyChanged(nameof(IsModelSection));
            OnPropertyChanged(nameof(IsTrainingSection));
            OnPropertyChanged(nameof(IsGenerationSection));
            OnPropertyChanged(nameof(IsGuideSection));
        }
    }

    public string DatasetText
    {
        get => _datasetText;
        set
        {
            if (!SetProperty(ref _datasetText, value)) return;
            RefreshStats();
            OnPropertyChanged(nameof(TokenizerIdealValuesText));
        }
    }

    public string DatasetPath
    {
        get => _datasetPath;
        set => SetProperty(ref _datasetPath, value);
    }

    public string PythonPath
    {
        get => _pythonPath;
        set
        {
            if (!SetProperty(ref _pythonPath, value)) return;
            OnPropertyChanged(nameof(BackendStatusText));
            OnPropertyChanged(nameof(IsBackendConfigured));
            _ = RefreshHardwareAsync();
        }
    }

    public string RunDirectory
    {
        get => _runDirectory;
        set
        {
            if (!SetProperty(ref _runDirectory, value)) return;
            OnPropertyChanged(nameof(RunDirectoryHelpText));
        }
    }

    public string CheckpointPath
    {
        get => _checkpointPath;
        set
        {
            if (!SetProperty(ref _checkpointPath, value)) return;
            _wizardCheckpointSet = !string.IsNullOrWhiteSpace(value);
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
        }
    }

    public string GenerationPrompt
    {
        get => _generationPrompt;
        set => SetProperty(ref _generationPrompt, value);
    }
    public string GeneratedText
    {
        get => _generatedText;
        private set => SetProperty(ref _generatedText, value);
    }
    public string GenerationStatusText
    {
        get => _generationStatusText;
        private set => SetProperty(ref _generationStatusText, value);
    }
    public bool IsGenerating
    {
        get => _isGenerating;
        private set
        {
            if (!SetProperty(ref _isGenerating, value)) return;
            OnPropertyChanged(nameof(IsGenerationIdle));
        }
    }
    public bool IsGenerationIdle => !IsGenerating;

    public bool IsTraining
    {
        get => _isTraining;
        set
        {
            if (!SetProperty(ref _isTraining, value)) return;
            OnPropertyChanged(nameof(IsTrainingIdle));
            OnPropertyChanged(nameof(TrainingStatusText));
        }
    }

    public bool ShowStartupOverlay
    {
        get => _showStartupOverlay;
        set => SetProperty(ref _showStartupOverlay, value);
    }
    public bool WizardEnabled
    {
        get => _wizardEnabled;
        set
        {
            if (!SetProperty(ref _wizardEnabled, value)) return;
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool ShowSettingsOverlay
    {
        get => _showSettingsOverlay;
        set
        {
            if (!SetProperty(ref _showSettingsOverlay, value)) return;
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool ShowAboutOverlay
    {
        get => _showAboutOverlay;
        set
        {
            if (!SetProperty(ref _showAboutOverlay, value)) return;
            OnPropertyChanged(nameof(ShowWizardOverlay));
        }
    }
    public bool IsEnglish
    {
        get => _isEnglish;
        set
        {
            if (!SetProperty(ref _isEnglish, value)) return;
            RefreshLocalizedTexts();
        }
    }
    public bool IsLightTheme
    {
        get => _isLightTheme;
        set => SetProperty(ref _isLightTheme, value);
    }

    public int BackendSetupProgress
    {
        get => _backendSetupProgress;
        set => SetProperty(ref _backendSetupProgress, value);
    }

    public string BackendSetupStage
    {
        get => _backendSetupStage;
        set => SetProperty(ref _backendSetupStage, value);
    }

    public TokenizerKind SelectedTokenizerKind
    {
        get => _selectedTokenizerKind;
        set
        {
            if (SetProperty(ref _selectedTokenizerKind, value))
            {
                TokenizerConfig.Kind = value;
                OnPropertyChanged(nameof(TokenizerExplanation));
                OnPropertyChanged(nameof(ShowBpeOptions));
                OnPropertyChanged(nameof(ShowWordOptions));
                OnPropertyChanged(nameof(ShowExperimentalOptions));
                ApplyRecommendedTokenizerSettings(value);
                ApplyRecommendedModelAndTrainingSettings(value);
                OnPropertyChanged(nameof(TokenizerConfig));
                OnPropertyChanged(nameof(TokenizerTargetVocabSize));
                OnPropertyChanged(nameof(TokenizerMinFrequency));
                OnPropertyChanged(nameof(TokenizerMaxMerges));
                OnPropertyChanged(nameof(TokenizerRecommendationText));
                OnPropertyChanged(nameof(TokenizerPresetBadgeText));
                OnPropertyChanged(nameof(TokenizerIdealValuesText));
                RaiseModelTrainingBindingsChanged();
                OnPropertyChanged(nameof(ParameterEstimate));
                RefreshWarnings();
            }
        }
    }

    public string Log
    {
        get => _log;
        set => SetProperty(ref _log, value);
    }

    public string TokenizerStatusText
    {
        get => _tokenizerStatusText;
        private set => SetProperty(ref _tokenizerStatusText, value);
    }

    public string BatchPreviewStatusText
    {
        get => _batchPreviewStatusText;
        private set => SetProperty(ref _batchPreviewStatusText, value);
    }

    public bool IsTokenizerReady => _lastTokenization is not null;
    public bool IsBatchPreviewReady => _wizardPreviewBuilt;
    public bool IsTrainingReady => _wizardDatasetImported && IsTokenizerReady && IsBatchPreviewReady;
    public string TrainingReadinessText
        => IsTrainingReady
            ? "Ready for training."
            : $"Missing steps: {string.Join(", ", GetTrainingMissingSteps())}.";

    public string DatasetStatsText { get; private set; } = string.Empty;
    public int TokenizerTargetVocabSize
    {
        get => TokenizerConfig.TargetVocabSize;
        set
        {
            if (TokenizerConfig.TargetVocabSize == value) return;
            TokenizerConfig.TargetVocabSize = value;
            OnPropertyChanged();
        }
    }
    public int TokenizerMinFrequency
    {
        get => TokenizerConfig.MinFrequency;
        set
        {
            if (TokenizerConfig.MinFrequency == value) return;
            TokenizerConfig.MinFrequency = value;
            OnPropertyChanged();
        }
    }
    public int TokenizerMaxMerges
    {
        get => TokenizerConfig.MaxMerges;
        set
        {
            if (TokenizerConfig.MaxMerges == value) return;
            TokenizerConfig.MaxMerges = value;
            OnPropertyChanged();
        }
    }
    public int ModelVocabSize
    {
        get => ModelConfig.VocabSize;
        set
        {
            if (ModelConfig.VocabSize == value) return;
            ModelConfig.VocabSize = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelBlockSize
    {
        get => ModelConfig.BlockSize;
        set
        {
            if (ModelConfig.BlockSize == value) return;
            ModelConfig.BlockSize = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelLayers
    {
        get => ModelConfig.Layers;
        set
        {
            if (ModelConfig.Layers == value) return;
            ModelConfig.Layers = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelHeads
    {
        get => ModelConfig.Heads;
        set
        {
            if (ModelConfig.Heads == value) return;
            ModelConfig.Heads = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int ModelEmbeddingSize
    {
        get => ModelConfig.EmbeddingSize;
        set
        {
            if (ModelConfig.EmbeddingSize == value) return;
            ModelConfig.EmbeddingSize = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ParameterEstimate));
            RefreshWarnings();
        }
    }
    public int TrainingBatchSize
    {
        get => TrainingConfig.BatchSize;
        set
        {
            if (TrainingConfig.BatchSize == value) return;
            TrainingConfig.BatchSize = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public int TrainingMaxSteps
    {
        get => TrainingConfig.MaxSteps;
        set
        {
            if (TrainingConfig.MaxSteps == value) return;
            TrainingConfig.MaxSteps = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public double TrainingLearningRate
    {
        get => TrainingConfig.LearningRate;
        set
        {
            if (Math.Abs(TrainingConfig.LearningRate - value) < 1e-12) return;
            TrainingConfig.LearningRate = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public int TrainingEvalEvery
    {
        get => TrainingConfig.EvalEvery;
        set
        {
            if (TrainingConfig.EvalEvery == value) return;
            TrainingConfig.EvalEvery = value;
            OnPropertyChanged();
            RefreshWarnings();
        }
    }
    public bool ShowAdvancedTraining
    {
        get => _showAdvancedTraining;
        set => SetProperty(ref _showAdvancedTraining, value);
    }
    public string SelectedTrainingProfile
    {
        get => _selectedTrainingProfile;
        set
        {
            if (!SetProperty(ref _selectedTrainingProfile, value)) return;
            ApplyTrainingProfile(value);
            RaiseModelTrainingBindingsChanged();
            RefreshWarnings();
            OnPropertyChanged(nameof(TrainingProfileHintText));
        }
    }
    public string TrainingProfileHintText => GuidedDefaultsEngine.DescribeTrainingProfile(SelectedTrainingProfile);
    public string TrainingOptimizer
    {
        get => TrainingConfig.Optimizer;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "adamw" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.Optimizer == next) return;
            TrainingConfig.Optimizer = next;
            OnPropertyChanged();
        }
    }
    public string TrainingScheduler
    {
        get => TrainingConfig.Scheduler;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "none" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.Scheduler == next) return;
            TrainingConfig.Scheduler = next;
            OnPropertyChanged();
        }
    }
    public int TrainingWarmupSteps
    {
        get => TrainingConfig.WarmupSteps;
        set
        {
            if (TrainingConfig.WarmupSteps == value) return;
            TrainingConfig.WarmupSteps = Math.Max(0, value);
            OnPropertyChanged();
        }
    }
    public bool TrainingMixedPrecision
    {
        get => TrainingConfig.MixedPrecision;
        set
        {
            if (TrainingConfig.MixedPrecision == value) return;
            TrainingConfig.MixedPrecision = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingPrecision
    {
        get => TrainingConfig.Precision;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "fp16" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.Precision == next) return;
            TrainingConfig.Precision = next;
            OnPropertyChanged();
        }
    }
    public bool TrainingGradientClipping
    {
        get => TrainingConfig.EnableGradientClipping;
        set
        {
            if (TrainingConfig.EnableGradientClipping == value) return;
            TrainingConfig.EnableGradientClipping = value;
            OnPropertyChanged();
        }
    }
    public double TrainingGradientClipNorm
    {
        get => TrainingConfig.GradientClipNorm;
        set
        {
            if (Math.Abs(TrainingConfig.GradientClipNorm - value) < 1e-12) return;
            TrainingConfig.GradientClipNorm = Math.Max(0.0, value);
            OnPropertyChanged();
        }
    }
    public int TrainingCheckpointEvery
    {
        get => TrainingConfig.CheckpointEvery;
        set
        {
            if (TrainingConfig.CheckpointEvery == value) return;
            TrainingConfig.CheckpointEvery = Math.Max(0, value);
            OnPropertyChanged();
        }
    }
    public bool TrainingEnablePostQuantization
    {
        get => TrainingConfig.EnablePostTrainingQuantization;
        set
        {
            if (TrainingConfig.EnablePostTrainingQuantization == value) return;
            TrainingConfig.EnablePostTrainingQuantization = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingQuantizationProfile
    {
        get => TrainingConfig.QuantizationProfile;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "dynamic-int8" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.QuantizationProfile == next) return;
            TrainingConfig.QuantizationProfile = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public int TrainingQuantizationCalibrationSamples
    {
        get => TrainingConfig.QuantizationCalibrationSamples;
        set
        {
            var next = Math.Max(1, value);
            if (TrainingConfig.QuantizationCalibrationSamples == next) return;
            TrainingConfig.QuantizationCalibrationSamples = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingEnableQatPath
    {
        get => TrainingConfig.EnableQatPath;
        set
        {
            if (TrainingConfig.EnableQatPath == value) return;
            TrainingConfig.EnableQatPath = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public int TrainingQatFineTuneSteps
    {
        get => TrainingConfig.QatFineTuneSteps;
        set
        {
            var next = Math.Max(1, value);
            if (TrainingConfig.QatFineTuneSteps == next) return;
            TrainingConfig.QatFineTuneSteps = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingDedup
    {
        get => TrainingConfig.EnableDeduplication;
        set
        {
            if (TrainingConfig.EnableDeduplication == value) return;
            TrainingConfig.EnableDeduplication = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingDedupLines
    {
        get => TrainingConfig.RemoveDuplicateLines;
        set
        {
            if (TrainingConfig.RemoveDuplicateLines == value) return;
            TrainingConfig.RemoveDuplicateLines = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingDedupParagraphs
    {
        get => TrainingConfig.RemoveDuplicateParagraphs;
        set
        {
            if (TrainingConfig.RemoveDuplicateParagraphs == value) return;
            TrainingConfig.RemoveDuplicateParagraphs = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingNormalizeUnicode
    {
        get => TrainingConfig.NormalizeUnicode;
        set
        {
            if (TrainingConfig.NormalizeUnicode == value) return;
            TrainingConfig.NormalizeUnicode = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingCollapseWhitespace
    {
        get => TrainingConfig.CollapseWhitespace;
        set
        {
            if (TrainingConfig.CollapseWhitespace == value) return;
            TrainingConfig.CollapseWhitespace = value;
            OnPropertyChanged();
        }
    }
    public bool TrainingCurriculumLearning
    {
        get => TrainingConfig.CurriculumLearning;
        set
        {
            if (TrainingConfig.CurriculumLearning == value) return;
            TrainingConfig.CurriculumLearning = value;
            OnPropertyChanged();
        }
    }
    public double TrainingCurriculumWarmupRatio
    {
        get => TrainingConfig.CurriculumWarmupRatio;
        set
        {
            if (Math.Abs(TrainingConfig.CurriculumWarmupRatio - value) < 1e-12) return;
            TrainingConfig.CurriculumWarmupRatio = Math.Clamp(value, 0.05, 0.9);
            OnPropertyChanged();
        }
    }
    public bool TrainingDistributed
    {
        get => TrainingConfig.DistributedTraining;
        set
        {
            if (TrainingConfig.DistributedTraining == value) return;
            TrainingConfig.DistributedTraining = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingClusterProfileName
    {
        get => TrainingConfig.ClusterProfileName;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "single-node" : value.Trim();
            if (TrainingConfig.ClusterProfileName == next) return;
            TrainingConfig.ClusterProfileName = next;
            var profile = ClusterProfileManager.Resolve(next);
            TrainingConfig.ClusterOrchestrator = profile.Orchestrator;
            TrainingConfig.ClusterWorldSize = Math.Max(1, profile.WorldSize);
            TrainingConfig.ClusterMaxRetries = Math.Max(0, profile.MaxRetries);
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
            OnPropertyChanged(nameof(TrainingWizardDetailedText));
        }
    }
    public bool TrainingOrchestratePipelineStages
    {
        get => TrainingConfig.OrchestratePipelineStages;
        set
        {
            if (TrainingConfig.OrchestratePipelineStages == value) return;
            TrainingConfig.OrchestratePipelineStages = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunDataStage
    {
        get => TrainingConfig.PipelineRunDataStage;
        set
        {
            if (TrainingConfig.PipelineRunDataStage == value) return;
            TrainingConfig.PipelineRunDataStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunPreprocessStage
    {
        get => TrainingConfig.PipelineRunPreprocessStage;
        set
        {
            if (TrainingConfig.PipelineRunPreprocessStage == value) return;
            TrainingConfig.PipelineRunPreprocessStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunTrainStage
    {
        get => TrainingConfig.PipelineRunTrainStage;
        set
        {
            if (TrainingConfig.PipelineRunTrainStage == value) return;
            TrainingConfig.PipelineRunTrainStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingPipelineRunEvalStage
    {
        get => TrainingConfig.PipelineRunEvalStage;
        set
        {
            if (TrainingConfig.PipelineRunEvalStage == value) return;
            TrainingConfig.PipelineRunEvalStage = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingMultiGpuStrategy
    {
        get => TrainingConfig.MultiGpuStrategy;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "none" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.MultiGpuStrategy == next) return;
            TrainingConfig.MultiGpuStrategy = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public int TrainingGradientAccumulationSteps
    {
        get => TrainingConfig.GradientAccumulationSteps;
        set
        {
            var next = Math.Max(1, value);
            if (TrainingConfig.GradientAccumulationSteps == next) return;
            TrainingConfig.GradientAccumulationSteps = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingAutoDeviceMap
    {
        get => TrainingConfig.AutoDeviceMap;
        set
        {
            if (TrainingConfig.AutoDeviceMap == value) return;
            TrainingConfig.AutoDeviceMap = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingAlignmentMode
    {
        get => TrainingConfig.AlignmentMode;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "none" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.AlignmentMode == next) return;
            TrainingConfig.AlignmentMode = next;
            OnPropertyChanged();
        }
    }
    public bool TrainingFineTuningOrchestration
    {
        get => TrainingConfig.FineTuningOrchestration;
        set
        {
            if (TrainingConfig.FineTuningOrchestration == value) return;
            TrainingConfig.FineTuningOrchestration = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingFineTuneStageSft
    {
        get => TrainingConfig.FineTuneStageSft;
        set
        {
            if (TrainingConfig.FineTuneStageSft == value) return;
            TrainingConfig.FineTuneStageSft = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingFineTuneStageDpo
    {
        get => TrainingConfig.FineTuneStageDpo;
        set
        {
            if (TrainingConfig.FineTuneStageDpo == value) return;
            TrainingConfig.FineTuneStageDpo = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingFineTuneStageRlhf
    {
        get => TrainingConfig.FineTuneStageRlhf;
        set
        {
            if (TrainingConfig.FineTuneStageRlhf == value) return;
            TrainingConfig.FineTuneStageRlhf = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingRlhfFeedbackSource
    {
        get => TrainingConfig.RlhfFeedbackSource;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "inline" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.RlhfFeedbackSource == next) return;
            TrainingConfig.RlhfFeedbackSource = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingRlhfFeedbackPath
    {
        get => TrainingConfig.RlhfFeedbackPath;
        set
        {
            if (TrainingConfig.RlhfFeedbackPath == value) return;
            TrainingConfig.RlhfFeedbackPath = value ?? string.Empty;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string RlhfDraftPrompt
    {
        get => _rlhfDraftPrompt;
        set => SetProperty(ref _rlhfDraftPrompt, value);
    }
    public string RlhfDraftChosen
    {
        get => _rlhfDraftChosen;
        set => SetProperty(ref _rlhfDraftChosen, value);
    }
    public string RlhfDraftRejected
    {
        get => _rlhfDraftRejected;
        set => SetProperty(ref _rlhfDraftRejected, value);
    }
    public string RlhfCollectedCountText => $"Collected RLHF records: {RlhfCollectedFeedback.Count}";
    public bool TrainingRewardModelingEnabled
    {
        get => TrainingConfig.RewardModelingEnabled;
        set
        {
            if (TrainingConfig.RewardModelingEnabled == value) return;
            TrainingConfig.RewardModelingEnabled = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingSafetyPolicyMode
    {
        get => TrainingConfig.SafetyPolicyMode;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "standard" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.SafetyPolicyMode == next) return;
            TrainingConfig.SafetyPolicyMode = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingExportOnnx
    {
        get => TrainingConfig.ExportOnnx;
        set
        {
            if (TrainingConfig.ExportOnnx == value) return;
            TrainingConfig.ExportOnnx = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public bool TrainingExportGguf
    {
        get => TrainingConfig.ExportGguf;
        set
        {
            if (TrainingConfig.ExportGguf == value) return;
            TrainingConfig.ExportGguf = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TrainingEvalSuite
    {
        get => TrainingConfig.EvalSuite;
        set
        {
            var next = string.IsNullOrWhiteSpace(value) ? "basic" : value.Trim().ToLowerInvariant();
            if (TrainingConfig.EvalSuite == next) return;
            TrainingConfig.EvalSuite = next;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string TokenizerExplanation => TokenizerRegistry.Explain(SelectedTokenizerKind);
    public bool ShowBpeOptions => SelectedTokenizerKind is TokenizerKind.SimpleBpe or TokenizerKind.ByteLevelBpe or TokenizerKind.Unigram or TokenizerKind.WordPiece or TokenizerKind.HybridFallback;
    public bool ShowWordOptions => SelectedTokenizerKind is TokenizerKind.Word;
    public bool ShowExperimentalOptions => SelectedTokenizerKind is TokenizerKind.HierarchicalExperimental;
    public bool IsDatasetSection => SelectedSection == "Dataset";
    public bool IsHardwareSection => SelectedSection == "Hardware";
    public bool IsTokenizationSection => SelectedSection == "Tokenization";
    public bool IsModelSection => SelectedSection == "Model";
    public bool IsTrainingSection => SelectedSection == "Training";
    public bool IsGenerationSection => SelectedSection == "Generation";
    public bool IsGuideSection => SelectedSection == "Guide";
    public string ParameterEstimate => ParameterEstimator.Human(ParameterEstimator.Estimate(ModelConfig));
    public string GenerationExplanation => GenerationPreviewService.Describe(SamplingConfig);
    public string ChartSummary => BuildChartSummary();
    public string TokenizerRecommendationText => BuildTokenizerRecommendationText();
    public string TokenizerPresetBadgeText => BuildTokenizerPresetBadgeText();
    public string TokenizerIdealValuesText => BuildTokenizerIdealValuesText();
    public string RoadmapChecklistText => BuildRoadmapChecklistText();
    public string HardwareCpuText { get; private set; } = "Detecting CPU...";
    public string HardwareRamText { get; private set; } = "Detecting RAM...";
    public string HardwareOsText { get; private set; } = "Detecting OS...";
    public string HardwareBackendText { get; private set; } = "Detecting backend...";
    public string HardwareNotesText { get; private set; } = string.Empty;
    public string HardwareAutoDetectText => IsEnglish
        ? "Automatic startup detection: CPU, RAM, GPU and backend device used for training/generation."
        : "Rilevamento automatico all'avvio: CPU, RAM, GPU e device backend usato per training/generation.";
    public string TokenizationStepGuide => IsEnglish
        ? "Recommended order: 1) Train Tokenizer  2) Build x/y Preview."
        : "Ordine consigliato: 1) Train Tokenizer  2) Build x/y Preview.";
    public string TrainingStepGuide => IsEnglish
        ? "Recommended order: 1) Verify readiness is complete  2) Start Backend Training  3) Monitor logs/loss  4) Cancel only if needed  5) Open Output Folder for checkpoints."
        : "Ordine consigliato: 1) Verifica readiness completa  2) Start Backend Training  3) Monitora log/loss  4) Cancel solo se necessario  5) Open Output Folder per checkpoint.";
    public string TrainingWizardMiniText => IsEnglish
        ? "Training Wizard: 1) Select training profile  2) Select eval pack  3) Check preflight status  4) Start training."
        : "Training Wizard: 1) Seleziona profilo training  2) Seleziona eval pack  3) Controlla preflight  4) Avvia training.";
    public string TrainingWizardDetailedText => BuildTrainingWizardDetailedText();
    public string TrainingPreflightText => BuildTrainingPreflightText();
    public string DatasetHelpText => IsEnglish
        ? "You can import a single file or a folder of files (.txt/.md/.csv), or paste text manually."
        : "Puoi importare singolo file o cartella di file (.txt/.md/.csv), oppure incollare testo manualmente.";
    public string RunDirectoryHelpText => IsEnglish
        ? "Output folder where JSONL logs, checkpoints and training manifest are saved."
        : "Cartella dove vengono salvati log JSONL, checkpoint e manifest del training.";
    public bool IsTrainingIdle => !IsTraining;
    public string TrainingStatusText
    {
        get => _trainingStatusText;
        private set => SetProperty(ref _trainingStatusText, value);
    }
    public bool TrainingForceCpu
    {
        get => TrainingConfig.ForceCpu;
        set
        {
            if (TrainingConfig.ForceCpu == value) return;
            TrainingConfig.ForceCpu = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(TrainingPreflightText));
        }
    }
    public string LastStepText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Last().Step.ToString("N0");
    public string LastTrainLossText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Last().TrainLoss.ToString("F6");
    public string LastValLossText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Last().ValLoss.ToString("F6");
    public string AvgTokensPerSecondText => TrainingLogs.Count == 0 ? "-" : TrainingLogs.Average(x => x.TokensPerSecond).ToString("F1");
    public string ElapsedTimeText
    {
        get
        {
            if (_trainingStartedAtUtc is null) return "-";
            var span = DateTimeOffset.UtcNow - _trainingStartedAtUtc.Value;
            if (span < TimeSpan.Zero) span = TimeSpan.Zero;
            return $"{(int)span.TotalMinutes:00}:{span.Seconds:00}";
        }
    }
    public string OverfittingStatusText => ComputeOverfittingStatus();
    public string TrainingQualityScoreText => ComputeTrainingQualityScore().score < 0 ? "-" : $"{ComputeTrainingQualityScore().score}/100";
    public string TrainingQualityColor => ComputeTrainingQualityScore().color;
    public string TrainingQualityBandText => ComputeTrainingQualityScore().band;
    public string TrainingQualityHintText => ComputeTrainingQualityScore().hint;
    public string EvalSuiteResultText => _lastEvalSummary is null
        ? "-"
        : $"{_lastEvalSummary.EvalSuite} ({_lastEvalSummary.NumBenchmarks} tasks)";
    public string EvalAverageScoreText => _lastEvalSummary is null ? "-" : _lastEvalSummary.AverageScore.ToString("F2");
    public string EvalBandText => _lastEvalSummary is null ? "-" : _lastEvalSummary.Band;
    public string EvalGateText => _lastEvalSummary is null ? "-" : (_lastEvalSummary.ReleaseGatePassed ? "PASS" : "FAIL");
    public string EvalGateThresholdText => _lastEvalSummary is null ? "-" : _lastEvalSummary.ReleaseGateThreshold.ToString("F1");
    public bool IsBackendBusy => _isBackendBootstrapping;
    public bool IsBackendConfigured => PythonBackendBridge.IsPythonAvailable(PythonPath);
    public bool ShowSetupBackendAction => !IsBackendConfigured;
    public string BackendStatusText => IsBackendBusy
        ? (_backendUserInitiatedStartup ? "Backend startup in progress..." : "Checking backend status...")
        : IsBackendConfigured
            ? $"Backend ready: {PythonPath}"
            : "Backend Python is not configured. Click 'Setup Backend' or set a valid interpreter path.";
    public double CpuUsage { get => _cpuUsage; set => SetProperty(ref _cpuUsage, value); }
    public double GpuUsage { get => _gpuUsage; set => SetProperty(ref _gpuUsage, value); }
    public double RamUsage { get => _ramUsage; set => SetProperty(ref _ramUsage, value); }
    public double DiskUsage { get => _diskUsage; set => SetProperty(ref _diskUsage, value); }
    public string CpuUsageText => $"{CpuUsage:0}%";
    public string GpuUsageText => $"{GpuUsage:0}%";
    public string RamUsageText => $"{RamUsage:0}%";
    public string DiskUsageText => $"{DiskUsage:0}%";
    public string WizardText => BuildWizardText();
    public string WizardProgressText => BuildWizardProgressText();
    public string WizardToggleText => IsEnglish ? "Guided Tour" : "Tour Guidato";
    public string SettingsTitle => IsEnglish ? "Settings" : "Impostazioni";
    public string AboutTitle => IsEnglish ? "About this software" : "Informazioni software";
    public string AboutText => IsEnglish
        ? "LLM Forge Studio\nDeveloped by Giovanni De Miccoli with OpenAI Codex support.\nDesktop application for local educational LLM experiments.\nVersion: 0.1.1\nLicense: MIT\nWebsite: www.giovannidemiccoli.it"
        : "LLM Forge Studio\nSviluppato da Giovanni De Miccoli con supporto OpenAI Codex.\nApplicazione desktop per esperimenti locali educativi su LLM.\nVersione: 0.1.1\nLicenza: MIT\nSito: www.giovannidemiccoli.it";
    public string LanguageLabel => IsEnglish ? "Language" : "Lingua";
    public string ThemeLabel => IsEnglish ? "Theme" : "Tema";
    public string LightLabel => IsEnglish ? "Light" : "Chiaro";
    public string DarkLabel => IsEnglish ? "Dark" : "Scuro";
    public string SectionWorkflowText => IsEnglish ? "Workflow" : "Workflow";
    public string HeaderSubtitleText => IsEnglish
        ? "Tokenizer, training loop, generation, and local LLM lab"
        : "Tokenizer, training loop, generation e laboratorio per LLM personali";
    public string ButtonLoadProject => IsEnglish ? "Load Project" : "Carica Progetto";
    public string ButtonSaveProject => IsEnglish ? "Save Project" : "Salva Progetto";
    public string ButtonGuide => IsEnglish ? "Guide" : "Guida";
    public string ButtonTrainTokenizer => IsEnglish ? "Train Tokenizer" : "Allena Tokenizer";
    public string WelcomeTitle => IsEnglish ? "Welcome to LLM Forge Studio" : "Benvenuto in LLM Forge Studio";
    public string GuideTitleText => IsEnglish ? "Guide - Page by Page" : "Guida - Pagina per Pagina";
    public string GuideIntroText => IsEnglish
        ? "Quick guide adapted from llm-from-scratch topics (excluding competition)."
        : "Guida rapida semplificata dalle parti 01-05 di llm-from-scratch (esclusa la competition).";
    public string GuideThanksText => IsEnglish
        ? "A special thank you to angelos-p for making this excellent guide available to everyone."
        : "Un ringraziamento speciale ad angelos-p per aver reso disponibile questa guida eccellente.";
    public string GuideDatasetText => IsEnglish
        ? "Goal: prepare text data. Import files/folder (.txt/.md/.csv), clean content, and validate dataset stats."
        : "Obiettivo: preparare il testo. Importa file o cartella (.txt/.md/.csv), pulisci e controlla le statistiche.";
    public string GuideTokenizationText => IsEnglish
        ? "Goal: convert text into token IDs. Run Train Tokenizer, then Build x/y Preview to validate next-token shift."
        : "Obiettivo: convertire testo in ID token. Prima Train Tokenizer, poi Build x/y Preview per verificare lo shift.";
    public string GuideTokenizationKeyText => IsEnglish
        ? "Key idea: smaller vocab on smaller datasets usually improves stability."
        : "Idea chiave: vocabolario piccolo su dataset piccolo evita sparsità e migliora la stabilità.";
    public string GuideModelText => IsEnglish
        ? "Goal: define Transformer capacity (block size, layers, heads, embedding) balancing quality and cost."
        : "Obiettivo: definire la capacità del Transformer bilanciando qualità e costi.";
    public string GuideModelKeyText => IsEnglish
        ? "Key idea: vocab size drives embeddings/output, while most params live in Transformer blocks."
        : "Idea chiave: il vocab size impatta embedding/output, mentre molti parametri sono nei blocchi Transformer.";
    public string GuideTrainingText => IsEnglish
        ? "Goal: learn next-token prediction. Start backend training and monitor train/val loss in real time."
        : "Obiettivo: apprendere next-token prediction. Avvia il backend training e monitora train/val loss.";
    public string GuideTrainingKeyText => IsEnglish
        ? "Key idea: if val loss rises while train loss improves, overfitting is likely."
        : "Idea chiave: se la val loss sale mentre train migliora, è probabile overfitting.";
    public string GuideGenerationText => IsEnglish
        ? "Goal: generate from checkpoint using prompt, temperature, top-k, and seed."
        : "Obiettivo: generare dal checkpoint usando prompt, temperature, top-k e seed.";
    public string GuideGenerationKeyText => IsEnglish
        ? "Key idea: evaluate quality frequently with real prompts, not metrics alone."
        : "Idea chiave: valuta spesso la qualità con prompt reali, non solo metriche.";
    public string GuideOrderText => IsEnglish
        ? "Recommended workflow: Dataset -> Tokenization -> Model -> Training -> Generation."
        : "Ordine operativo consigliato: Dataset -> Tokenization -> Model -> Training -> Generation.";
    public string GuideSourceText => IsEnglish
        ? "Source: github.com/angelos-p/llm-from-scratch/docs."
        : "Fonte: github.com/angelos-p/llm-from-scratch/docs.";
    public bool ShowWizardOverlay => WizardEnabled && !ShowSettingsOverlay && !ShowAboutOverlay;
    public string HeaderBackground => IsLightTheme ? "#E5E7EB" : "#0B1220";
    public string SidebarBackground => IsLightTheme ? "#F3F4F6" : "#0B1220";
    public string DividerBrush => IsLightTheme ? "#CBD5E1" : "#1F2A3A";
    public string ModalBackground => IsLightTheme ? "#FFFFFF" : "#0F172A";
    public string ModalBorderBrush => IsLightTheme ? "#CBD5E1" : "#334155";

    public void ExportProjectToLog()
    {
        var payload = BuildProjectPayload();
        Log = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
    }

    public async Task SaveProjectAsync(string path)
    {
        var project = BuildProjectPayload();
        await ProjectStore.SaveAsync(project, path);
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "project.save",
            "Project saved from UI.",
            new { path, snapshot = project });
        Log = $"Project saved: {path}";
    }

    public async Task LoadProjectAsync(string path)
    {
        var project = await ProjectStore.LoadAsync(path);
        DatasetPath = project.DatasetPath;
        DatasetText = project.DatasetText;
        ResetPreparationState();
        _wizardDatasetImported = !string.IsNullOrWhiteSpace(DatasetText);

        ApplyCleaner(project.Cleaner);
        ApplyTokenizer(project.Tokenizer);
        ApplyModel(project.Model);
        ApplyTraining(project.Training);
        ApplySampling(project.Sampling);

        SelectedSection = Sections.Contains(project.SelectedSection) ? project.SelectedSection : "Dataset";
        PythonPath = project.PythonPath;
        RunDirectory = project.RunDirectory;
        CheckpointPath = project.CheckpointPath;
        GenerationPrompt = project.GenerationPrompt;

        RefreshAll();
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "project.load",
            "Project loaded from UI.",
            new { path, snapshot = BuildProjectPayload() });
        Log = $"Project loaded: {path}";
    }

    public async Task ImportDatasetFromPathAsync(string path)
    {
        var cleaned = await DatasetLoader.LoadTextAsync(path, Cleaner);
        DatasetPath = path;
        DatasetText = cleaned;
        ResetPreparationState();
        _wizardDatasetImported = true;
        Log = $"Dataset imported: {Path.GetFileName(path)}";
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
    }

    public async Task ImportDatasetFromFolderAsync(string folderPath)
    {
        if (!Directory.Exists(folderPath))
        {
            Log = $"Cartella non trovata: {folderPath}";
            return;
        }

        var files = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly)
            .Where(p => p.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".md", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".csv", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
                     || p.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();

        if (files.Count == 0)
        {
            Log = "Nessun file supportato trovato nella cartella (.txt/.md/.csv/.jsonl/.json).";
            return;
        }

        var blocks = new List<string>(files.Count);
        foreach (var file in files)
        {
            var cleaned = await DatasetLoader.LoadTextAsync(file, Cleaner);
            blocks.Add(cleaned);
        }

        DatasetPath = folderPath;
        DatasetText = string.Join("\n\n", blocks);
        ResetPreparationState();
        _wizardDatasetImported = true;

        var rlhfCandidate = ResolveRlhfFeedbackPathCandidate(files);
        if (!string.IsNullOrWhiteSpace(rlhfCandidate))
        {
            TrainingConfig.RlhfFeedbackPath = rlhfCandidate;
            OnPropertyChanged(nameof(TrainingRlhfFeedbackPath));
            Log = $"Dataset cartella importato: {files.Count} file uniti.\nAuto-detect RLHF feedback path: {rlhfCandidate}";
        }
        else
        {
            Log = $"Dataset cartella importato: {files.Count} file uniti.";
        }

        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
    }

    private static string ResolveRlhfFeedbackPathCandidate(IReadOnlyList<string> files)
    {
        var jsonlFiles = files
            .Where(p => p.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => p)
            .ToList();
        if (jsonlFiles.Count == 0) return string.Empty;
        if (jsonlFiles.Count == 1) return jsonlFiles[0];

        var heuristic = jsonlFiles.FirstOrDefault(p =>
        {
            var name = Path.GetFileName(p).ToLowerInvariant();
            return name.Contains("rlhf")
                || name.Contains("feedback")
                || name.Contains("human");
        });
        return heuristic ?? string.Empty;
    }

    private void RefreshAll()
    {
        RefreshStats();
        RefreshWarnings();
        OnPropertyChanged(nameof(ParameterEstimate));
        OnPropertyChanged(nameof(GenerationExplanation));
        OnPropertyChanged(nameof(ChartSummary));
        OnPropertyChanged(nameof(TokenizerRecommendationText));
        OnPropertyChanged(nameof(BackendStatusText));
        OnPropertyChanged(nameof(IsBackendConfigured));
        OnPropertyChanged(nameof(ShowSetupBackendAction));
        OnPropertyChanged(nameof(RunDirectoryHelpText));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
    }

    public async Task RefreshHardwareAsync()
    {
        var root = ResolveProjectRoot();
        var summary = await HardwareProbe.DetectAsync(root, PythonPath);
        HardwareCpuText = $"{summary.CpuName} ({summary.LogicalCores} logical cores)";
        HardwareRamText = summary.TotalRamGb > 0 ? $"{summary.TotalRamGb:F1} GB RAM" : "RAM non rilevata";
        HardwareOsText = summary.OsDescription;
        HardwareBackendText = summary.BackendDeviceStatus;
        HardwareNotesText = summary.BackendNotes;

        HardwareGpus.Clear();
        foreach (var gpu in summary.Gpus) HardwareGpus.Add(gpu);

        OnPropertyChanged(nameof(HardwareCpuText));
        OnPropertyChanged(nameof(HardwareRamText));
        OnPropertyChanged(nameof(HardwareOsText));
        OnPropertyChanged(nameof(HardwareBackendText));
        OnPropertyChanged(nameof(HardwareNotesText));
    }

    private void RefreshStats()
    {
        var stats = TextCleaner.Analyze(DatasetText);
        DatasetStatsText = $"Chars: {stats.CharacterCount:N0} | Lines: {stats.LineCount:N0} | Words: {stats.ApproxWordCount:N0} | Unique chars: {stats.UniqueCharacterCount:N0}";
        OnPropertyChanged(nameof(DatasetStatsText));
    }

    private void TrainTokenizer()
    {
        if (string.IsNullOrWhiteSpace(DatasetText))
        {
            Log = "Cannot train tokenizer: dataset text is empty.";
            return;
        }

        var tokenizer = TokenizerRegistry.Create(SelectedTokenizerKind);
        tokenizer.Train(DatasetText, TokenizerConfig);
        var ids = tokenizer.Encode(DatasetText);
        var decoded = tokenizer.Decode(ids);
        _lastTokenization = new TokenizationResult { TokenIds = ids, Vocabulary = tokenizer.Vocabulary, DecodedPreview = decoded };
        _wizardTokenizerTrained = true;

        VocabularyPreview.Clear();
        foreach (var item in tokenizer.Vocabulary.Take(200)) VocabularyPreview.Add(item);

        ModelConfig.VocabSize = Math.Max(1, tokenizer.Vocabulary.Count);
        Log = $"Tokenizer trained: {tokenizer.Name}. Vocab={tokenizer.Vocabulary.Count}, Tokens={ids.Count}.\nDecoded preview:\n{decoded}";
        TokenizerStatusText = $"Tokenizer ready: {tokenizer.Name}, vocab {tokenizer.Vocabulary.Count:N0}, tokens {ids.Count:N0}.";
        BatchPreviewStatusText = "x/y preview not built yet.";
        RaiseModelTrainingBindingsChanged();
        RefreshWarnings();
        OnPropertyChanged(nameof(ParameterEstimate));
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
        OnPropertyChanged(nameof(IsTokenizerReady));
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
    }

    private void BuildBatchPreview()
    {
        if (_lastTokenization is null) TrainTokenizer();
        if (_lastTokenization is null) return;

        var preview = TrainingBatchBuilder.BuildPreview(_lastTokenization.TokenIds, ModelConfig.BlockSize);
        _wizardPreviewBuilt = true;
        Log = preview.Explanation + "\n\nX:\n" + string.Join(", ", preview.X.Take(64)) + "\n\nY:\n" + string.Join(", ", preview.Y.Take(64));
        BatchPreviewStatusText = $"x/y preview ready: X={preview.X.Count:N0}, Y={preview.Y.Count:N0}, showing first 64 IDs.";
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));
        OnPropertyChanged(nameof(IsBatchPreviewReady));
        OnPropertyChanged(nameof(IsTrainingReady));
        OnPropertyChanged(nameof(TrainingReadinessText));
    }

    private void RunDryTraining()
    {
        if (_lastTokenization is null) TrainTokenizer();
        TrainingLogs.Clear();
        foreach (var entry in DryRunTrainer.Simulate(TrainingConfig, _lastTokenization?.TokenCount ?? 0))
            TrainingLogs.Add(entry);
        Log = "Dry-run completed.";
        OnPropertyChanged(nameof(ChartSummary));
    }

    private async Task StartBackendTrainingAsync()
    {
        if (IsTraining) return;
        if (!IsTrainingReady)
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.missing_steps",
                "Training blocked because required preparation steps are missing.",
                new { missing = GetTrainingMissingSteps(), snapshot = BuildProjectPayload() });
            Log = $"Training blocked. Missing steps: {string.Join(", ", GetTrainingMissingSteps())}.";
            return;
        }
        if (!PythonBackendBridge.IsPythonAvailable(PythonPath))
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.python",
                "Training blocked because configured Python executable is unavailable.",
                new { pythonPath = PythonPath, snapshot = BuildProjectPayload() });
            Log = $"Python non trovato: {PythonPath}";
            return;
        }
        var preflightBlock = GetBlockingPreflightIssue();
        if (!string.IsNullOrWhiteSpace(preflightBlock))
        {
            TrainingStatusText = "Training blocked by preflight checks.";
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.blocked.preflight",
                "Training blocked by preflight checks.",
                new { preflightBlock, snapshot = BuildProjectPayload() });
            Log = preflightBlock;
            return;
        }

        var projectRoot = ResolveProjectRoot();
        var scriptPath = Path.Combine(projectRoot, "backends", "python", "cluster_runner.py");
        await PrepareRlhfFeedbackForRunAsync();
        var datasetForBackend = await WriteConsolidatedDatasetForBackendAsync();
        var spec = new BackendJobSpec
        {
            JobType = "train",
            DatasetPath = datasetForBackend,
            OutputDirectory = RunDirectory,
            Tokenizer = TokenizerConfig,
            Model = ModelConfig,
            Training = TrainingConfig,
            Sampling = SamplingConfig,
            ClusterProfileName = TrainingConfig.ClusterProfileName
        };

        var specPath = await PythonBackendBridge.WriteJobSpecAsync(spec, RunDirectory);
        var clusterDescriptor = ClusterJobDescriptor.FromSpec(spec);
        clusterDescriptor.JobSpecPath = specPath;
        var clusterDescriptorPath = await ClusterJobDescriptor.WriteAsync(RunDirectory, clusterDescriptor);
        spec.ClusterJobDescriptorPath = clusterDescriptorPath;
        specPath = await PythonBackendBridge.WriteJobSpecAsync(spec, RunDirectory);
        var startInfo = PythonBackendBridge.CreateStartInfo(PythonPath, scriptPath, $"--job \"{specPath}\"");
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.start.requested",
            "Training process requested from UI.",
            new
            {
                runDirectory = RunDirectory,
                projectRoot,
                scriptPath,
                specPath,
                clusterDescriptorPath,
                snapshot = BuildProjectPayload()
            });

        _trainingProcess = Process.Start(startInfo);
        if (_trainingProcess is null)
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.start.failed",
                "Backend process failed to start.",
                new { startInfo.FileName, startInfo.Arguments, runDirectory = RunDirectory });
            Log = "Errore avvio processo training.";
            TrainingStatusText = "Training failed: backend process could not start.";
            return;
        }

        var stdOutTask = _trainingProcess.StandardOutput.ReadToEndAsync();
        var stdErrTask = _trainingProcess.StandardError.ReadToEndAsync();

        IsTraining = true;
        _trainingStartedAtUtc = DateTimeOffset.UtcNow;
        _wizardTrainingStarted = true;
        TrainingLogs.Clear();
        TrainingStatusText = "Training started. Waiting for backend logs...";
        Log = $"Training started. Output directory: {RunDirectory}";
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.started",
            "Training process started.",
            new { processId = _trainingProcess.Id, runDirectory = RunDirectory });
        RaiseTrainingDashboardChanged();
        OnPropertyChanged(nameof(WizardText));
        OnPropertyChanged(nameof(WizardProgressText));

        _ = MonitorBackendTrainingAsync(_trainingProcess, stdOutTask, stdErrTask);
    }

    private async Task MonitorBackendTrainingAsync(Process process, Task<string> stdOutTask, Task<string> stdErrTask)
    {
        try
        {
            while (!process.HasExited)
            {
                await PollTrainingLogAsync();
                if (TrainingLogs.Count == 0)
                {
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        TrainingStatusText = "Training running... waiting for first log entry.";
                    });
                }
                await Task.Delay(400);
            }

            await PollTrainingLogAsync();

            var stdOut = await stdOutTask;
            var stdErr = await stdErrTask;
            var exitCode = process.ExitCode;

            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                var modelPath = Path.Combine(RunDirectory, "model.pt");
                var manifestPath = Path.Combine(RunDirectory, "checkpoint_manifest.json");
                var lastStep = TrainingLogs.Count > 0 ? TrainingLogs.Last().Step : -1;
                var hasArtifacts = File.Exists(modelPath) && File.Exists(manifestPath);
                var likelyCompleted = hasArtifacts && lastStep >= TrainingConfig.MaxSteps;

                if (exitCode == 0)
                {
                    _ = RunArtifactRegistry.UpdateAsync(RunDirectory);
                    _ = RefreshEvalSummaryAsync();
                    TrainingStatusText = $"Training completed successfully. Steps logged: {TrainingLogs.Count}.";
                    Log = $"Training completed. Output: {RunDirectory}";
                    _ = _uiDebugLogger.WriteAsync(
                        ResolveRunDirectoryForDebug(),
                        "training.completed.success",
                        "Training completed successfully.",
                        new { exitCode, runDirectory = RunDirectory, modelPath, manifestPath, lastStep, hasArtifacts });
                    AutoSelectCheckpointIfAvailable();
                }
                else if (likelyCompleted)
                {
                    _ = RunArtifactRegistry.UpdateAsync(RunDirectory);
                    _ = RefreshEvalSummaryAsync();
                    TrainingStatusText = $"Training completed (with non-zero exit code {exitCode}). Artifacts are valid.";
                    Log = $"Training completed with warning (exit code {exitCode}) but checkpoint artifacts were produced.\nOutput: {RunDirectory}";
                    _ = _uiDebugLogger.WriteAsync(
                        ResolveRunDirectoryForDebug(),
                        "training.completed.warning",
                        "Training exited with non-zero code but artifacts look valid.",
                        new { exitCode, runDirectory = RunDirectory, modelPath, manifestPath, lastStep, hasArtifacts });
                    AutoSelectCheckpointIfAvailable();
                }
                else
                {
                    var err = string.IsNullOrWhiteSpace(stdErr) ? stdOut : stdErr;
                    var compactErr = string.IsNullOrWhiteSpace(err) ? "(no backend error output)" : err.Trim();
                    TrainingStatusText = $"Training failed (exit code {exitCode}).";
                    Log = $"Training failed (exit code {exitCode}).\nBackend output:\n{compactErr}";
                    _ = _uiDebugLogger.WriteAsync(
                        ResolveRunDirectoryForDebug(),
                        "training.completed.failed",
                        "Training failed.",
                        new { exitCode, runDirectory = RunDirectory, backendError = compactErr, modelPath, manifestPath, lastStep, hasArtifacts });
                }
            });
        }
        catch (Exception ex)
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                TrainingStatusText = "Training monitor failed.";
                Log = $"Training monitor error: {ex.Message}";
            });
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.monitor.error",
                "Training monitor threw an exception.",
                new { exception = ex.ToString(), runDirectory = RunDirectory });
        }
        finally
        {
            await Dispatcher.UIThread.InvokeAsync(() =>
            {
                IsTraining = false;
                RaiseTrainingDashboardChanged();
            });
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "training.monitor.closed",
                "Training monitor closed.",
                new { runDirectory = RunDirectory, logEntries = TrainingLogs.Count });
        }
    }

    private string[] GetTrainingMissingSteps()
    {
        var missing = new List<string>();
        if (!_wizardDatasetImported) missing.Add("import dataset");
        if (!IsTokenizerReady) missing.Add("train tokenizer");
        if (!IsBatchPreviewReady) missing.Add("build x/y preview");
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

    private async Task<string> WriteConsolidatedDatasetForBackendAsync()
    {
        var raw = DatasetText ?? string.Empty;
        return await FineTuningStageOrchestrator.PrepareAsync(raw, TrainingConfig, RunDirectory);
    }

    private async Task PrepareRlhfFeedbackForRunAsync()
    {
        if (!TrainingConfig.FineTuneStageRlhf) return;
        if (!string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase)) return;
        if (RlhfCollectedFeedback.Count == 0) return;

        var path = Path.Combine(RunDirectory, "rlhf_feedback_collected.jsonl");
        await RlhfFeedbackCollector.SaveJsonlAsync(RlhfCollectedFeedback, path);
        TrainingConfig.RlhfFeedbackPath = path;
        OnPropertyChanged(nameof(TrainingRlhfFeedbackPath));
    }

    private void AddRlhfFeedback()
    {
        var prompt = (RlhfDraftPrompt ?? string.Empty).Trim();
        var chosen = (RlhfDraftChosen ?? string.Empty).Trim();
        var rejected = (RlhfDraftRejected ?? string.Empty).Trim();
        if (prompt.Length == 0 || chosen.Length == 0)
        {
            Log = "RLHF feedback requires prompt and chosen response.";
            return;
        }

        RlhfCollectedFeedback.Add(new RlhfFeedbackRecord
        {
            Prompt = prompt,
            Chosen = chosen,
            Rejected = rejected
        });

        RlhfDraftPrompt = string.Empty;
        RlhfDraftChosen = string.Empty;
        RlhfDraftRejected = string.Empty;
        OnPropertyChanged(nameof(RlhfCollectedCountText));
    }

    private void ClearRlhfFeedback()
    {
        RlhfCollectedFeedback.Clear();
        OnPropertyChanged(nameof(RlhfCollectedCountText));
    }

    private async Task SetupBackendAsync()
    {
        ShowStartupOverlay = false;
        await EnsureBackendReadyAsync(forceInstall: false, userInitiated: true);
        await RefreshHardwareAsync();
    }

    private async Task PollTrainingLogAsync()
    {
        var logPath = Path.Combine(RunDirectory, "train_log.jsonl");
        if (!File.Exists(logPath)) return;

        string[] lines;
        try
        {
            await using var fs = new FileStream(logPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var sr = new StreamReader(fs);
            var content = await sr.ReadToEndAsync();
            lines = content.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
        }
        catch (IOException)
        {
            // Writer still holds/updates the file; skip this poll cycle.
            return;
        }

        var parsed = new List<TrainingLogEntry>();
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            // Defensive parse: runtime/OS can occasionally inject non-JSON bytes while file is being written.
            var cleanLine = line.Trim().TrimStart('\0', '\uFEFF');
            if (cleanLine.Length == 0) continue;

            try
            {
                using var doc = JsonDocument.Parse(cleanLine);
                var root = doc.RootElement;
                parsed.Add(new TrainingLogEntry(
                    root.GetProperty("step").GetInt32(),
                    root.GetProperty("train_loss").GetDouble(),
                    root.GetProperty("val_loss").GetDouble(),
                    root.GetProperty("tokens_per_second").GetDouble(),
                    root.TryGetProperty("message", out var message) ? message.GetString() ?? string.Empty : string.Empty));
            }
            catch (JsonException)
            {
                // Skip malformed partial line and continue polling.
            }
        }

        await Dispatcher.UIThread.InvokeAsync(() =>
        {
            TrainingLogs.Clear();
            foreach (var item in parsed.TakeLast(500)) TrainingLogs.Add(item);
            if (TrainingLogs.Count > 0)
            {
                var last = TrainingLogs.Last();
                TrainingStatusText = $"Training running... step {last.Step}, train loss {last.TrainLoss:F4}, val loss {last.ValLoss:F4}.";
            }
            RaiseTrainingDashboardChanged();
            OnPropertyChanged(nameof(ChartSummary));
        });
    }

    private void CancelBackendTraining()
    {
        if (_trainingProcess is null || _trainingProcess.HasExited) return;
        _trainingProcess.Kill(entireProcessTree: true);
        Log = "Training cancelled by user.";
        TrainingStatusText = "Training cancelled.";
        IsTraining = false;
        _ = _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "training.cancelled",
            "Training cancelled by user.",
            new { runDirectory = RunDirectory, processId = _trainingProcess.Id });
        RaiseTrainingDashboardChanged();
    }

    private void AutoSelectCheckpointIfAvailable()
    {
        var manifest = Path.Combine(RunDirectory, "checkpoint_manifest.json");
        if (!File.Exists(manifest)) return;
        CheckpointPath = manifest;
        _wizardCheckpointSet = true;
    }

    private string ComputeOverfittingStatus()
    {
        if (TrainingLogs.Count < 4) return "Overfitting signal: not enough validation points yet.";

        var valSeries = TrainingLogs.Select(x => x.ValLoss).ToList();
        var trainSeries = TrainingLogs.Select(x => x.TrainLoss).ToList();
        var bestVal = valSeries.Min();
        var currentVal = valSeries[^1];
        var previousVal = valSeries[^2];
        var currentTrain = trainSeries[^1];
        var previousTrain = trainSeries[^2];

        var valWorsening = currentVal > previousVal && currentVal > bestVal * 1.10;
        var trainImproving = currentTrain <= previousTrain;
        if (valWorsening && trainImproving)
            return "Overfitting signal: warning (val loss worsening while train loss improves).";

        if (currentVal <= bestVal * 1.02)
            return "Overfitting signal: healthy (validation near best).";

        return "Overfitting signal: monitor (no strong overfitting pattern yet).";
    }

    private (int score, string color, string band, string hint) ComputeTrainingQualityScore()
    {
        if (TrainingLogs.Count < 4)
            return (-1, "#94A3B8", "Not enough data", "Run longer training and collect more validation checkpoints.");

        var logs = TrainingLogs.ToList();
        var first = logs.First();
        var last = logs.Last();
        var bestVal = logs.Min(x => x.ValLoss);
        var valGain = first.ValLoss - last.ValLoss;
        var trainValGap = Math.Abs(last.TrainLoss - last.ValLoss);
        var overfitPenalty = last.ValLoss > bestVal * 1.08 && last.TrainLoss < last.ValLoss ? 18 : 0;
        var gainScore = Math.Clamp((int)Math.Round(valGain * 40.0), 0, 45);
        var stabilityScore = Math.Clamp((int)Math.Round((1.0 - Math.Min(trainValGap, 1.0)) * 35.0), 0, 35);
        var speedScore = Math.Clamp((int)Math.Round(Math.Min(logs.Average(x => x.TokensPerSecond) / 200000.0, 1.0) * 20.0), 0, 20);
        var score = Math.Clamp(gainScore + stabilityScore + speedScore - overfitPenalty, 0, 100);

        var band = "Poor";
        var color = "#EF4444";
        if (score >= 80) { band = "Excellent"; color = "#22C55E"; }
        else if (score >= 60) { band = "Good"; color = "#84CC16"; }
        else if (score >= 40) { band = "Fair"; color = "#F59E0B"; }

        var hint = score switch
        {
            < 40 => "Low score: increase dataset size/quality, reduce learning rate, and raise eval frequency.",
            < 60 => "Fair score: try lower learning rate and longer training with more regular validation checks.",
            < 80 => "Good score: tune block size and layers gradually, then compare validation trend.",
            _ => "Excellent score: keep settings stable and validate with new prompts/checkpoints."
        };

        return (score, color, band, hint);
    }

    private void RaiseTrainingDashboardChanged()
    {
        OnPropertyChanged(nameof(LastStepText));
        OnPropertyChanged(nameof(LastTrainLossText));
        OnPropertyChanged(nameof(LastValLossText));
        OnPropertyChanged(nameof(AvgTokensPerSecondText));
        OnPropertyChanged(nameof(ElapsedTimeText));
        OnPropertyChanged(nameof(OverfittingStatusText));
        OnPropertyChanged(nameof(TrainingQualityScoreText));
        OnPropertyChanged(nameof(TrainingQualityColor));
        OnPropertyChanged(nameof(TrainingQualityBandText));
        OnPropertyChanged(nameof(TrainingQualityHintText));
        OnPropertyChanged(nameof(EvalSuiteResultText));
        OnPropertyChanged(nameof(EvalAverageScoreText));
        OnPropertyChanged(nameof(EvalBandText));
        OnPropertyChanged(nameof(EvalGateText));
        OnPropertyChanged(nameof(EvalGateThresholdText));
    }

    private async Task RefreshEvalSummaryAsync()
    {
        var snapshot = await EvalSummaryReader.TryReadAsync(RunDirectory);
        await Dispatcher.UIThread.InvokeAsync(() =>
        {
            _lastEvalSummary = snapshot;
            RaiseTrainingDashboardChanged();
        });
    }

    private static async Task RunProcessAsync(string fileName, string arguments, string workingDirectory)
    {
        var info = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        using var process = Process.Start(info);
        if (process is null) throw new InvalidOperationException($"Impossibile avviare processo: {fileName}");

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(string.IsNullOrWhiteSpace(error) ? output : error);
    }

    private async Task BootstrapBackendOnStartupAsync()
    {
        await EnsureBackendReadyAsync(forceInstall: false, userInitiated: false);
        await RefreshHardwareAsync();
        ShowStartupOverlay = false;
    }

    private async Task EnsureBackendReadyAsync(bool forceInstall, bool userInitiated)
    {
        if (_isBackendBootstrapping)
        {
            Log = IsEnglish ? "Backend setup is already running. Please wait." : "Setup backend già in corso, attendi il completamento...";
            return;
        }
        _isBackendBootstrapping = true;
        _backendUserInitiatedStartup = userInitiated;
        BackendSetupProgress = 0;
        BackendSetupStage = userInitiated ? "Backend startup in progress..." : "Checking backend status...";
        OnPropertyChanged(nameof(IsBackendBusy));
        OnPropertyChanged(nameof(BackendStatusText));
        try
        {
            var root = ResolveProjectRoot();
            var backendDir = Path.Combine(root, "backends", "python");
            if (!Directory.Exists(backendDir))
            {
                Log = IsEnglish ? $"Backend directory not found: {backendDir}" : $"Backend directory non trovata: {backendDir}";
                return;
            }

            var venvPython = OperatingSystem.IsWindows()
                ? Path.Combine(backendDir, ".venv", "Scripts", "python.exe")
                : Path.Combine(backendDir, ".venv", "bin", "python");

            var bundledPython = OperatingSystem.IsWindows()
                ? Path.Combine(root, "runtime", "python", "python.exe")
                : Path.Combine(root, "runtime", "python", "bin", "python3");

            string? bootstrapPython = null;
            if (!forceInstall && PythonBackendBridge.IsPythonAvailable(venvPython))
            {
                PythonPath = venvPython;
                _wizardSetupDone = true;
                SetBackendProgress(100, "Backend ready");
                Log = IsEnglish ? $"Backend ready (local venv): {venvPython}" : $"Backend pronto (venv locale): {venvPython}";
                return;
            }

            if (PythonBackendBridge.IsPythonAvailable(bundledPython))
                bootstrapPython = bundledPython;
            else if (PythonBackendBridge.IsPythonAvailable("python"))
                bootstrapPython = "python";
            else if (PythonBackendBridge.IsPythonAvailable("py"))
                bootstrapPython = "py";

            if (bootstrapPython is null)
            {
                SetBackendProgress(0, "Backend not configured");
                Log = IsEnglish
                    ? "Python was not found. Add embedded runtime in runtime/python or install system Python."
                    : "Python non trovato. Inserisci runtime embedded in runtime/python oppure installa Python di sistema.";
                return;
            }

            if (!File.Exists(venvPython))
            {
                SetBackendProgress(10, "Creating virtual environment...");
                Log = IsEnglish ? "Automatic backend setup: creating virtual environment..." : "Configurazione backend automatica: creazione virtualenv...";
                await RunProcessAsync(bootstrapPython, "-m venv .venv", backendDir);
            }
            SetBackendProgress(25, "Installing dependencies...");
            Log = IsEnglish ? "Automatic backend setup: installing dependencies..." : "Configurazione backend automatica: installazione dipendenze...";
            await InstallBackendDependenciesAsync(venvPython, backendDir);
            PythonPath = venvPython;
            _wizardSetupDone = true;
            SetBackendProgress(95, "Verifying backend...");
            await RunProcessAsync(venvPython, "-c \"import torch; print('ok')\"", backendDir);
            SetBackendProgress(100, "Backend ready");
            Log = IsEnglish ? $"Backend configured automatically: {venvPython}" : $"Backend configurato automaticamente: {venvPython}";
        }
        catch (Exception ex)
        {
            SetBackendProgress(0, IsEnglish ? "Setup failed" : "Setup fallito");
            Log = IsEnglish ? $"Backend bootstrap failed: {ex.Message}" : $"Bootstrap backend fallito: {ex.Message}";
        }
        finally
        {
            _isBackendBootstrapping = false;
            _backendUserInitiatedStartup = false;
            OnPropertyChanged(nameof(BackendStatusText));
            OnPropertyChanged(nameof(IsBackendConfigured));
            OnPropertyChanged(nameof(ShowSetupBackendAction));
            OnPropertyChanged(nameof(IsBackendBusy));
            OnPropertyChanged(nameof(WizardText));
            OnPropertyChanged(nameof(WizardProgressText));
        }
    }

    private async Task InstallBackendDependenciesAsync(string venvPython, string backendDir)
    {
        var flavor = DetectBackendFlavor();

        SetBackendProgress(35, IsEnglish ? "Updating pip..." : "Aggiornamento pip...");
        await RunProcessAsync(venvPython, "-m pip install --upgrade pip", backendDir);
        SetBackendProgress(45, IsEnglish ? "Installing requirements..." : "Installazione requirements...");
        await RunProcessAsync(venvPython, "-m pip install -r requirements.txt", backendDir);
        SetBackendProgress(55, IsEnglish ? "Cleaning old torch packages..." : "Pulizia vecchi pacchetti torch...");
        await RunProcessAsync(venvPython, "-m pip uninstall -y torch torchvision torchaudio torch-directml", backendDir);

        switch (flavor)
        {
            case BackendFlavor.NvidiaCuda:
                SetBackendProgress(70, IsEnglish ? "Installing PyTorch CUDA (NVIDIA)..." : "Installazione PyTorch CUDA (NVIDIA)...");
                Log = IsEnglish ? "NVIDIA GPU detected: installing PyTorch CUDA..." : "GPU NVIDIA rilevata: installo PyTorch CUDA...";
                await RunProcessAsync(
                    venvPython,
                    "-m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio",
                    backendDir);
                break;
            case BackendFlavor.AmdDirectMl:
                SetBackendProgress(70, IsEnglish ? "Installing PyTorch (AMD)..." : "Installazione PyTorch (AMD)...");
                Log = IsEnglish ? "AMD GPU detected: installing PyTorch + DirectML..." : "GPU AMD rilevata: installo PyTorch + DirectML...";
                await RunProcessAsync(venvPython, "-m pip install --upgrade torch torchvision torchaudio", backendDir);
                SetBackendProgress(82, IsEnglish ? "Installing DirectML..." : "Installazione DirectML...");
                await RunProcessAsync(venvPython, "-m pip install --upgrade torch-directml", backendDir);
                break;
            default:
                SetBackendProgress(70, IsEnglish ? "Installing PyTorch CPU..." : "Installazione PyTorch CPU...");
                Log = IsEnglish ? "No compatible GPU detected: installing PyTorch CPU fallback..." : "Nessuna GPU compatibile rilevata: installo PyTorch CPU fallback...";
                await RunProcessAsync(venvPython, "-m pip install --upgrade torch torchvision torchaudio", backendDir);
                break;
        }
        SetBackendProgress(90, IsEnglish ? "Finalizing..." : "Finalizzazione...");
    }

    private static BackendFlavor DetectBackendFlavor()
    {
        try
        {
            if (!OperatingSystem.IsWindows()) return BackendFlavor.Cpu;

            var info = new ProcessStartInfo
            {
                FileName = "cmd",
                Arguments = "/c wmic path win32_VideoController get Name",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var process = Process.Start(info);
            if (process is null) return BackendFlavor.Cpu;
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(5000);

            if (output.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase)) return BackendFlavor.NvidiaCuda;
            if (output.Contains("AMD", StringComparison.OrdinalIgnoreCase) || output.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                return BackendFlavor.AmdDirectMl;
        }
        catch
        {
            // ignored, fallback to CPU
        }

        return BackendFlavor.Cpu;
    }

    private async Task RefreshPerformanceAsync()
    {
        if (_isPerfSampling) return;
        _isPerfSampling = true;
        try
        {
            var values = await Task.Run(() => (
                cpu: ReadCpuUsage(),
                gpu: ReadGpuUsage(),
                ram: ReadRamUsage(),
                disk: ReadDiskUsage()
            ));

            CpuUsage = values.cpu;
            GpuUsage = values.gpu;
            RamUsage = values.ram;
            DiskUsage = values.disk;
            OnPropertyChanged(nameof(CpuUsageText));
            OnPropertyChanged(nameof(GpuUsageText));
            OnPropertyChanged(nameof(RamUsageText));
            OnPropertyChanged(nameof(DiskUsageText));
        }
        finally
        {
            _isPerfSampling = false;
        }
    }

    private static double ReadCpuUsage()
    {
        var output = RunCommand("cmd", "/c wmic cpu get loadpercentage /value");
        var line = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("LoadPercentage=", StringComparison.OrdinalIgnoreCase));
        if (line is null) return 0;
        return double.TryParse(line.Split('=', 2)[1].Trim(), out var value) ? Math.Clamp(value, 0, 100) : 0;
    }

    private static double ReadGpuUsage()
    {
        var output = RunCommand("cmd", "/c nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits");
        var line = output.Split('\n').Select(x => x.Trim()).FirstOrDefault(x => !string.IsNullOrWhiteSpace(x));
        if (line is null) return 0;
        if (double.TryParse(line, out var value)) return Math.Clamp(value, 0, 100);
        return 0;
    }

    private static double ReadRamUsage()
    {
        var output = RunCommand("cmd", "/c wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /value");
        var freeLine = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("FreePhysicalMemory=", StringComparison.OrdinalIgnoreCase));
        var totalLine = output.Split('\n').FirstOrDefault(l => l.TrimStart().StartsWith("TotalVisibleMemorySize=", StringComparison.OrdinalIgnoreCase));
        if (freeLine is null || totalLine is null) return 0;
        if (!double.TryParse(freeLine.Split('=', 2)[1].Trim(), out var freeKb)) return 0;
        if (!double.TryParse(totalLine.Split('=', 2)[1].Trim(), out var totalKb)) return 0;
        var used = Math.Max(0, totalKb - freeKb);
        return Math.Clamp(used / Math.Max(1, totalKb) * 100.0, 0, 100);
    }

    private static double ReadDiskUsage()
    {
        var root = Path.GetPathRoot(Directory.GetCurrentDirectory()) ?? "C:\\";
        var drive = new DriveInfo(root);
        var used = drive.TotalSize - drive.TotalFreeSpace;
        return Math.Clamp((double)used / Math.Max(1, drive.TotalSize) * 100.0, 0, 100);
    }

    private string BuildWizardText()
    {
        if (!WizardEnabled) return IsEnglish ? "Guided tour disabled." : "Tour guidato disattivato.";
        if (!_wizardSetupDone)
            return IsEnglish
                ? "Step 1/6 - Hardware: click Setup Backend. Why: the app installs the right runtime (NVIDIA CUDA / AMD DirectML / CPU fallback) so training can run."
                : "Step 1/6 - Hardware: clicca Setup Backend. Perché: l'app installa il runtime corretto (NVIDIA CUDA / AMD DirectML / CPU fallback) per poter fare training.";

        if (!_wizardDatasetImported)
            return IsEnglish
                ? "Step 2/6 - Dataset: import file/folder first. Why: model quality depends directly on data quality and size."
                : "Step 2/6 - Dataset: importa prima file/cartella. Perché: qualità e dimensione del dato determinano la qualità del modello.";

        if (!_wizardTokenizerTrained)
            return IsEnglish
                ? "Step 3/6 - Tokenization: run Train Tokenizer first. Why: it builds the vocabulary and converts text into token IDs."
                : "Step 3/6 - Tokenization: esegui prima Train Tokenizer. Perché: costruisce il vocabolario e converte il testo in token ID.";

        if (!_wizardPreviewBuilt)
            return IsEnglish
                ? "Step 4/6 - Tokenization check: now click Build x/y Preview. Why: verify next-token shift (Y is X shifted by +1) before training."
                : "Step 4/6 - Controllo tokenization: ora clicca Build x/y Preview. Perché: verifichi lo shift next-token (Y è X spostato di +1) prima del training.";

        if (!_wizardTrainingStarted)
            return IsEnglish
                ? "Step 5/6 - Training: click Start Backend Training. Why: this creates checkpoint artifacts used by generation."
                : "Step 5/6 - Training: clicca Start Backend Training. Perché: crea i file checkpoint necessari alla generazione.";

        if (!_wizardCheckpointSet)
            return IsEnglish
                ? "Step 6/6 - Generation setup: set checkpoint_manifest.json path in Generation."
                : "Step 6/6 - Setup generation: imposta il path di checkpoint_manifest.json in Generation.";

        return IsEnglish
            ? "Step 6/6 - Generate: run Generate from Checkpoint, then iterate temperature/top-k/seed. Why: this is where you evaluate coherence and style."
            : "Step 6/6 - Generate: esegui Generate from Checkpoint, poi itera temperature/top-k/seed. Perché: qui valuti coerenza e stile del modello.";
    }

    private string BuildWizardProgressText()
    {
        if (!WizardEnabled) return IsEnglish ? "Tour off" : "Tour off";
        if (!_wizardSetupDone) return "1/6";
        if (!_wizardDatasetImported) return "2/6";
        if (!_wizardTokenizerTrained) return "3/6";
        if (!_wizardPreviewBuilt) return "4/6";
        if (!_wizardTrainingStarted) return "5/6";
        if (!_wizardCheckpointSet) return "6/6";
        return "6/6";
    }

    private void SetLanguage(bool english)
    {
        IsEnglish = english;
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
        OnPropertyChanged(nameof(ButtonGuide));
        OnPropertyChanged(nameof(ButtonTrainTokenizer));
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
        OnPropertyChanged(nameof(GuideOrderText));
        OnPropertyChanged(nameof(GuideSourceText));
        OnPropertyChanged(nameof(HardwareAutoDetectText));
        OnPropertyChanged(nameof(TokenizationStepGuide));
        OnPropertyChanged(nameof(TrainingStepGuide));
        OnPropertyChanged(nameof(TrainingWizardMiniText));
        OnPropertyChanged(nameof(TrainingPreflightText));
        OnPropertyChanged(nameof(DatasetHelpText));
        OnPropertyChanged(nameof(RunDirectoryHelpText));
    }

    private void SetBackendProgress(int value, string stage)
    {
        BackendSetupProgress = Math.Clamp(value, 0, 100);
        BackendSetupStage = stage;
        Log = $"[{BackendSetupProgress}%] {stage}";
    }

    private void GeneratePreview()
    {
        var fakeLogits = Enumerable.Range(0, Math.Max(1, ModelConfig.VocabSize)).Select(i => Math.Sin(i * 0.37) * 3.0).ToArray();
        var chosen = Sampler.Sample(fakeLogits, SamplingConfig);
        Log = $"Generation preview. {GenerationExplanation}\nChosen token id from fake logits: {chosen}";
    }

    private async Task GenerateFromCheckpointAsync()
    {
        if (IsGenerating) return;
        if (!PythonBackendBridge.IsPythonAvailable(PythonPath))
        {
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "generation.blocked.python",
                "Generation blocked because configured Python executable is unavailable.",
                new { pythonPath = PythonPath, checkpointPath = CheckpointPath, prompt = GenerationPrompt });
            Log = $"Python non trovato: {PythonPath}";
            return;
        }

        IsGenerating = true;
        GeneratedText = string.Empty;
        GenerationStatusText = "Thinking...";

        var projectRoot = ResolveProjectRoot();
        var scriptPath = Path.Combine(projectRoot, "backends", "python", "generate_stub.py");
        await _uiDebugLogger.WriteAsync(
            ResolveRunDirectoryForDebug(),
            "generation.start.requested",
            "Generation from checkpoint requested from UI.",
            new
            {
                runDirectory = RunDirectory,
                checkpointPath = CheckpointPath,
                prompt = GenerationPrompt,
                sampling = Clone(SamplingConfig),
                snapshot = BuildProjectPayload()
            });
        var startInfo = PythonBackendBridge.CreateStartInfo(
            PythonPath,
            scriptPath,
            $"--checkpoint \"{CheckpointPath}\" --prompt \"{GenerationPrompt.Replace("\"", "\\\"")}\" --temperature {SamplingConfig.Temperature.ToString(CultureInfo.InvariantCulture)} --top-k {SamplingConfig.TopK} --seed {SamplingConfig.Seed} --max-new-tokens {SamplingConfig.MaxNewTokens}");

        try
        {
            using var process = Process.Start(startInfo);
            if (process is null)
            {
                await _uiDebugLogger.WriteAsync(
                    ResolveRunDirectoryForDebug(),
                    "generation.start.failed",
                    "Generation backend process failed to start.",
                    new { startInfo.FileName, startInfo.Arguments, runDirectory = RunDirectory });
                Log = "Errore avvio generation backend.";
                GenerationStatusText = "Generation failed: backend process could not start.";
                return;
            }

            var output = await process.StandardOutput.ReadToEndAsync();
            var err = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode != 0)
            {
                await _uiDebugLogger.WriteAsync(
                    ResolveRunDirectoryForDebug(),
                    "generation.completed.failed",
                    "Generation failed.",
                    new { exitCode = process.ExitCode, stderr = err, checkpointPath = CheckpointPath, prompt = GenerationPrompt });
                Log = $"Generation fallita: {err}";
                GenerationStatusText = "Generation failed.";
                return;
            }

            using var doc = JsonDocument.Parse(output);
            var text = doc.RootElement.GetProperty("text").GetString() ?? string.Empty;
            await StreamGeneratedTextAsync(text);
            GenerationStatusText = $"Done. Generated with max new tokens = {SamplingConfig.MaxNewTokens}.";
            await _uiDebugLogger.WriteAsync(
                ResolveRunDirectoryForDebug(),
                "generation.completed.success",
                "Generation completed successfully.",
                new
                {
                    checkpointPath = CheckpointPath,
                    prompt = GenerationPrompt,
                    outputPreview = text.Length > 500 ? text[..500] : text
                });
            Log = "Generation completed successfully.";
        }
        finally
        {
            IsGenerating = false;
        }
    }

    private async Task StreamGeneratedTextAsync(string fullText)
    {
        var parts = fullText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0)
        {
            GeneratedText = string.Empty;
            return;
        }

        var sb = new StringBuilder();
        foreach (var part in parts)
        {
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(part);
            var current = sb.ToString();
            await Dispatcher.UIThread.InvokeAsync(() => { GeneratedText = current; });
            await Task.Delay(18);
        }
    }

    private void RefreshWarnings()
    {
        Warnings.Clear();
        var datasetChars = DatasetText.Length;
        var tokenCount = _lastTokenization?.TokenCount ?? 0;
        foreach (var w in TokenizerCompatibility.Validate(TokenizerConfig, datasetChars, ModelConfig.BlockSize)) Warnings.Add(w);
        foreach (var w in CompatibilityRules.Validate(ModelConfig, TrainingConfig, tokenCount)) Warnings.Add(w);

        if (TrainingConfig.DistributedTraining)
        {
            var worldSizeRaw = Environment.GetEnvironmentVariable("WORLD_SIZE");
            var hasWorldSize = int.TryParse(worldSizeRaw, out var worldSize) && worldSize > 1;
            if (!hasWorldSize)
                Warnings.Add("Distributed mode is enabled but WORLD_SIZE is not set (>1). Run may fallback to single-device behavior.");
        }

        if (TrainingConfig.EnablePostTrainingQuantization && !TrainingConfig.ForceCpu)
            Warnings.Add("Post-training quantization currently applies on CPU path. Enable Force CPU if you want quantized artifact generation.");

        if (TrainingConfig.MixedPrecision && TrainingConfig.ForceCpu)
            Warnings.Add("Mixed precision is enabled with Force CPU. AMP acceleration is typically effective only on CUDA devices.");

        if (!Warnings.Any()) Warnings.Add("No major compatibility warnings.");
    }

    private ForgeProject BuildProjectPayload() => new()
    {
        DatasetPath = DatasetPath,
        DatasetText = DatasetText,
        Cleaner = Clone(Cleaner),
        Tokenizer = Clone(TokenizerConfig),
        Model = Clone(ModelConfig),
        Training = Clone(TrainingConfig),
        Sampling = Clone(SamplingConfig),
        SelectedSection = SelectedSection,
        PythonPath = PythonPath,
        RunDirectory = RunDirectory,
        CheckpointPath = CheckpointPath,
        GenerationPrompt = GenerationPrompt
    };

    private static T Clone<T>(T value)
    {
        var json = JsonSerializer.Serialize(value);
        return JsonSerializer.Deserialize<T>(json)!;
    }

    private string ResolveRunDirectoryForDebug()
        => string.IsNullOrWhiteSpace(RunDirectory)
            ? Path.Combine(Environment.CurrentDirectory, "runs", "default")
            : RunDirectory;

    private void ApplyCleaner(TextCleanerConfig config)
    {
        Cleaner.NormalizeNewLines = config.NormalizeNewLines;
        Cleaner.TrimLines = config.TrimLines;
        Cleaner.Lowercase = config.Lowercase;
        Cleaner.RemoveEmptyLines = config.RemoveEmptyLines;
    }

    private void ApplyTokenizer(TokenizerConfig config)
    {
        TokenizerConfig.Kind = config.Kind;
        TokenizerConfig.LowercaseBeforeTraining = config.LowercaseBeforeTraining;
        TokenizerConfig.TargetVocabSize = config.TargetVocabSize;
        TokenizerConfig.MinFrequency = config.MinFrequency;
        TokenizerConfig.MaxMerges = config.MaxMerges;
        TokenizerConfig.KeepPunctuationAsTokens = config.KeepPunctuationAsTokens;
        TokenizerConfig.UseCharacterFallback = config.UseCharacterFallback;
        TokenizerConfig.Notes = config.Notes;
        SelectedTokenizerKind = config.Kind;
    }

    private void ApplyModel(ModelConfig config)
    {
        ModelConfig.VocabSize = config.VocabSize;
        ModelConfig.BlockSize = config.BlockSize;
        ModelConfig.Layers = config.Layers;
        ModelConfig.Heads = config.Heads;
        ModelConfig.EmbeddingSize = config.EmbeddingSize;
        ModelConfig.Dropout = config.Dropout;
        RaiseModelTrainingBindingsChanged();
    }

    private void ApplyTraining(TrainingConfig config)
    {
        TrainingConfig.BatchSize = config.BatchSize;
        TrainingConfig.MaxSteps = config.MaxSteps;
        TrainingConfig.LearningRate = config.LearningRate;
        TrainingConfig.EvalEvery = config.EvalEvery;
        TrainingConfig.TrainSplit = config.TrainSplit;
        TrainingConfig.EnableGradientClipping = config.EnableGradientClipping;
        TrainingConfig.GradientClipNorm = config.GradientClipNorm;
        TrainingConfig.Optimizer = config.Optimizer;
        TrainingConfig.Scheduler = config.Scheduler;
        TrainingConfig.WarmupSteps = config.WarmupSteps;
        TrainingConfig.MixedPrecision = config.MixedPrecision;
        TrainingConfig.Precision = config.Precision;
        TrainingConfig.CheckpointEvery = config.CheckpointEvery;
        TrainingConfig.EnablePostTrainingQuantization = config.EnablePostTrainingQuantization;
        TrainingConfig.QuantizationProfile = config.QuantizationProfile;
        TrainingConfig.QuantizationCalibrationSamples = config.QuantizationCalibrationSamples;
        TrainingConfig.EnableQatPath = config.EnableQatPath;
        TrainingConfig.QatFineTuneSteps = Math.Max(1, config.QatFineTuneSteps);
        TrainingConfig.EnableDeduplication = config.EnableDeduplication;
        TrainingConfig.RemoveDuplicateLines = config.RemoveDuplicateLines;
        TrainingConfig.RemoveDuplicateParagraphs = config.RemoveDuplicateParagraphs;
        TrainingConfig.NormalizeUnicode = config.NormalizeUnicode;
        TrainingConfig.CollapseWhitespace = config.CollapseWhitespace;
        TrainingConfig.CurriculumLearning = config.CurriculumLearning;
        TrainingConfig.CurriculumWarmupRatio = config.CurriculumWarmupRatio;
        TrainingConfig.ResumeDatasetFromState = config.ResumeDatasetFromState;
        TrainingConfig.DeterministicShardShuffle = config.DeterministicShardShuffle;
        TrainingConfig.DataShuffleSeed = config.DataShuffleSeed;
        TrainingConfig.ClusterProfileName = config.ClusterProfileName;
        TrainingConfig.ClusterOrchestrator = config.ClusterOrchestrator;
        TrainingConfig.ClusterWorldSize = config.ClusterWorldSize;
        TrainingConfig.ClusterMaxRetries = config.ClusterMaxRetries;
        TrainingConfig.ClusterHeartbeatSeconds = config.ClusterHeartbeatSeconds;
        TrainingConfig.OrchestratePipelineStages = config.OrchestratePipelineStages;
        TrainingConfig.PipelineRunDataStage = config.PipelineRunDataStage;
        TrainingConfig.PipelineRunPreprocessStage = config.PipelineRunPreprocessStage;
        TrainingConfig.PipelineRunTrainStage = config.PipelineRunTrainStage;
        TrainingConfig.PipelineRunEvalStage = config.PipelineRunEvalStage;
        TrainingConfig.DistributedTraining = config.DistributedTraining;
        TrainingConfig.MultiGpuStrategy = string.IsNullOrWhiteSpace(config.MultiGpuStrategy) ? "none" : config.MultiGpuStrategy;
        TrainingConfig.GradientAccumulationSteps = Math.Max(1, config.GradientAccumulationSteps);
        TrainingConfig.AutoDeviceMap = config.AutoDeviceMap;
        TrainingConfig.AlignmentMode = config.AlignmentMode;
        TrainingConfig.FineTuningOrchestration = config.FineTuningOrchestration;
        TrainingConfig.FineTuneStageSft = config.FineTuneStageSft;
        TrainingConfig.FineTuneStageDpo = config.FineTuneStageDpo;
        TrainingConfig.FineTuneStageRlhf = config.FineTuneStageRlhf;
        TrainingConfig.RlhfFeedbackSource = string.IsNullOrWhiteSpace(config.RlhfFeedbackSource) ? "inline" : config.RlhfFeedbackSource;
        TrainingConfig.RlhfFeedbackPath = config.RlhfFeedbackPath ?? string.Empty;
        TrainingConfig.RewardModelingEnabled = config.RewardModelingEnabled;
        TrainingConfig.SafetyPolicyMode = string.IsNullOrWhiteSpace(config.SafetyPolicyMode) ? "standard" : config.SafetyPolicyMode;
        TrainingConfig.ExportOnnx = config.ExportOnnx;
        TrainingConfig.ExportGguf = config.ExportGguf;
        TrainingConfig.EvalSuite = config.EvalSuite;
        TrainingConfig.ForceCpu = config.ForceCpu;
        _selectedTrainingProfile = "Custom";
        OnPropertyChanged(nameof(SelectedTrainingProfile));
        RaiseModelTrainingBindingsChanged();
    }

    private void ApplySampling(SamplingConfig config)
    {
        SamplingConfig.Temperature = config.Temperature;
        SamplingConfig.TopK = config.TopK;
        SamplingConfig.Seed = config.Seed;
        SamplingConfig.Greedy = config.Greedy;
        SamplingConfig.MaxNewTokens = config.MaxNewTokens;
    }

    private string BuildChartSummary()
    {
        if (TrainingLogs.Count == 0) return "No chart data yet.";
        var last = TrainingLogs.Last();
        var avgTok = TrainingLogs.Average(x => x.TokensPerSecond);
        return $"Last step={last.Step} | train={last.TrainLoss:F4} | val={last.ValLoss:F4} | tok/s(avg)={avgTok:F1}";
    }

    private void ApplyRecommendedTokenizerSettings(TokenizerKind kind)
    {
        GuidedDefaultsEngine.ApplyTokenizerPreset(kind, DatasetText, TokenizerConfig);
    }

    private void ApplyRecommendedModelAndTrainingSettings(TokenizerKind kind)
    {
        GuidedDefaultsEngine.ApplyModelTrainingPresetForTokenizer(kind, ModelConfig, TrainingConfig);
        RaiseModelTrainingBindingsChanged();
    }

    private void ApplyTrainingProfile(string profile)
    {
        GuidedDefaultsEngine.ApplyTrainingProfile(profile, TrainingConfig);
    }

    private void RaiseModelTrainingBindingsChanged()
    {
        OnPropertyChanged(nameof(ModelVocabSize));
        OnPropertyChanged(nameof(ModelBlockSize));
        OnPropertyChanged(nameof(ModelLayers));
        OnPropertyChanged(nameof(ModelHeads));
        OnPropertyChanged(nameof(ModelEmbeddingSize));
        OnPropertyChanged(nameof(TrainingBatchSize));
        OnPropertyChanged(nameof(TrainingMaxSteps));
        OnPropertyChanged(nameof(TrainingLearningRate));
        OnPropertyChanged(nameof(TrainingEvalEvery));
        OnPropertyChanged(nameof(TrainingForceCpu));
        OnPropertyChanged(nameof(TrainingOptimizer));
        OnPropertyChanged(nameof(TrainingScheduler));
        OnPropertyChanged(nameof(TrainingWarmupSteps));
        OnPropertyChanged(nameof(TrainingMixedPrecision));
        OnPropertyChanged(nameof(TrainingPrecision));
        OnPropertyChanged(nameof(TrainingGradientClipping));
        OnPropertyChanged(nameof(TrainingGradientClipNorm));
        OnPropertyChanged(nameof(TrainingCheckpointEvery));
        OnPropertyChanged(nameof(TrainingEnablePostQuantization));
        OnPropertyChanged(nameof(TrainingQuantizationProfile));
        OnPropertyChanged(nameof(TrainingQuantizationCalibrationSamples));
        OnPropertyChanged(nameof(TrainingEnableQatPath));
        OnPropertyChanged(nameof(TrainingQatFineTuneSteps));
        OnPropertyChanged(nameof(TrainingDedup));
        OnPropertyChanged(nameof(TrainingDedupLines));
        OnPropertyChanged(nameof(TrainingDedupParagraphs));
        OnPropertyChanged(nameof(TrainingNormalizeUnicode));
        OnPropertyChanged(nameof(TrainingCollapseWhitespace));
        OnPropertyChanged(nameof(TrainingCurriculumLearning));
        OnPropertyChanged(nameof(TrainingCurriculumWarmupRatio));
        OnPropertyChanged(nameof(TrainingOrchestratePipelineStages));
        OnPropertyChanged(nameof(TrainingPipelineRunDataStage));
        OnPropertyChanged(nameof(TrainingPipelineRunPreprocessStage));
        OnPropertyChanged(nameof(TrainingPipelineRunTrainStage));
        OnPropertyChanged(nameof(TrainingPipelineRunEvalStage));
        OnPropertyChanged(nameof(TrainingClusterProfileName));
        OnPropertyChanged(nameof(TrainingDistributed));
        OnPropertyChanged(nameof(TrainingMultiGpuStrategy));
        OnPropertyChanged(nameof(TrainingGradientAccumulationSteps));
        OnPropertyChanged(nameof(TrainingAutoDeviceMap));
        OnPropertyChanged(nameof(TrainingAlignmentMode));
        OnPropertyChanged(nameof(TrainingFineTuningOrchestration));
        OnPropertyChanged(nameof(TrainingFineTuneStageSft));
        OnPropertyChanged(nameof(TrainingFineTuneStageDpo));
        OnPropertyChanged(nameof(TrainingFineTuneStageRlhf));
        OnPropertyChanged(nameof(TrainingRlhfFeedbackSource));
        OnPropertyChanged(nameof(TrainingRlhfFeedbackPath));
        OnPropertyChanged(nameof(TrainingRewardModelingEnabled));
        OnPropertyChanged(nameof(TrainingSafetyPolicyMode));
        OnPropertyChanged(nameof(TrainingExportOnnx));
        OnPropertyChanged(nameof(TrainingExportGguf));
        OnPropertyChanged(nameof(TrainingEvalSuite));
        OnPropertyChanged(nameof(TrainingWizardDetailedText));
        OnPropertyChanged(nameof(TrainingPreflightText));
        OnPropertyChanged(nameof(ParameterEstimate));
    }

    private string BuildTokenizerRecommendationText()
    {
        return SelectedTokenizerKind switch
        {
            TokenizerKind.Character => "Preset Character: vocab 256, min freq 1, merges 0. Utile per dataset piccolo o testi rumorosi.",
            TokenizerKind.Word => "Preset Word: vocab proporzionale al dataset, min freq 2, merges 0. Più leggibile ma più OOV su parole nuove.",
            TokenizerKind.ByteLevelBpe => "Preset Byte-level BPE: robusto su UTF-8 e simboli, con merges medi/alti per corpus reali.",
            TokenizerKind.Unigram => "Preset Unigram: subword probabilistico, efficace su corpora vari e testi misti.",
            TokenizerKind.WordPiece => "Preset WordPiece: subword con prefisso ##, utile per pipeline di fine-tuning NLP.",
            TokenizerKind.SimpleBpe => "Preset BPE: vocab e merges medi, min freq 2. Buon compromesso tra robustezza e compressione.",
            TokenizerKind.HybridFallback => "Preset Hybrid: come BPE ma con fallback char per OOV. Più robusto in generazione reale.",
            TokenizerKind.HierarchicalExperimental => "Preset Experimental: parametri alti per esplorazione macro/subword; non consigliato per produzione.",
            _ => string.Empty
        };
    }

    private string BuildTokenizerPresetBadgeText()
    {
        return SelectedTokenizerKind switch
        {
            TokenizerKind.Character => "Mode: Educational",
            TokenizerKind.Word => "Mode: Basic NLP",
            TokenizerKind.SimpleBpe => "Mode: Balanced",
            TokenizerKind.ByteLevelBpe => "Mode: Robust UTF-8",
            TokenizerKind.Unigram => "Mode: Probabilistic Subword",
            TokenizerKind.WordPiece => "Mode: Fine-tuning Ready",
            TokenizerKind.HybridFallback => "Mode: Production-safe Local",
            TokenizerKind.HierarchicalExperimental => "Mode: Experimental",
            _ => "Mode: Custom"
        };
    }

    private string BuildTokenizerIdealValuesText()
    {
        var probe = Clone(TokenizerConfig);
        GuidedDefaultsEngine.ApplyTokenizerPreset(SelectedTokenizerKind, DatasetText, probe);
        var fallback = probe.UseCharacterFallback ? "on" : "off";
        return $"Ideal values: vocab {probe.TargetVocabSize:N0}, min freq {probe.MinFrequency}, merges {probe.MaxMerges:N0}, char fallback {fallback}.";
    }

    private string BuildRoadmapChecklistText()
    {
        return string.Join('\n',
            "Roadmap Checklist (in-app):",
            "- Advanced Training panel: completed/verified",
            "- Guided defaults engine: completed/verified",
            "- Advanced tokenizer options: byte-level BPE + unigram + wordpiece",
            "- Multi-GPU/cluster/RLHF: planned next blocks");
    }

    private string BuildTrainingPreflightText()
    {
        var notes = new List<string>();

        if (TrainingConfig.DistributedTraining)
        {
            var wsRaw = Environment.GetEnvironmentVariable("WORLD_SIZE");
            if (!int.TryParse(wsRaw, out var ws) || ws <= 1)
                notes.Add("Distributed ON: set WORLD_SIZE>1 before start.");
            else
                notes.Add($"Distributed ON: WORLD_SIZE={ws}.");
            notes.Add($"Multi-GPU strategy: {TrainingConfig.MultiGpuStrategy}.");
        }
        else
        {
            notes.Add("Distributed OFF: single-device mode.");
        }
        if (TrainingConfig.OrchestratePipelineStages)
        {
            notes.Add($"Pipeline ON: data={TrainingConfig.PipelineRunDataStage}, preprocess={TrainingConfig.PipelineRunPreprocessStage}, train={TrainingConfig.PipelineRunTrainStage}, eval={TrainingConfig.PipelineRunEvalStage}.");
        }
        else
        {
            notes.Add("Pipeline OFF: single train stage.");
        }

        notes.Add($"Grad accumulation: x{Math.Max(1, TrainingConfig.GradientAccumulationSteps)}.");
        notes.Add($"Auto device map: {(TrainingConfig.AutoDeviceMap ? "ON" : "OFF")}.");
        if (TrainingConfig.FineTuningOrchestration)
            notes.Add($"Fine-tuning pipeline ON: sft={TrainingConfig.FineTuneStageSft}, dpo={TrainingConfig.FineTuneStageDpo}, rlhf={TrainingConfig.FineTuneStageRlhf}.");
        else
            notes.Add($"Fine-tuning pipeline OFF: alignment mode={TrainingConfig.AlignmentMode}.");
        if (TrainingConfig.FineTuneStageRlhf && string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase))
            notes.Add($"RLHF inline feedback collected: {RlhfCollectedFeedback.Count}.");
        notes.Add($"Reward model: {(TrainingConfig.RewardModelingEnabled ? "ON" : "OFF")}.");
        notes.Add($"Safety policy: {TrainingConfig.SafetyPolicyMode}.");
        notes.Add($"Export targets: onnx={TrainingConfig.ExportOnnx}, gguf={TrainingConfig.ExportGguf}.");
        notes.Add($"Cluster profile: {TrainingConfig.ClusterProfileName} ({TrainingConfig.ClusterOrchestrator}, world={TrainingConfig.ClusterWorldSize}).");

        notes.Add($"Eval pack: {TrainingConfig.EvalSuite}");

        if (TrainingConfig.EnablePostTrainingQuantization && !TrainingConfig.ForceCpu)
            notes.Add("Quantization ON but Force CPU OFF (artifact may be skipped).");
        if (TrainingConfig.EnablePostTrainingQuantization)
            notes.Add($"Quant profile: {TrainingConfig.QuantizationProfile}, calib={TrainingConfig.QuantizationCalibrationSamples}.");
        if (TrainingConfig.EnableQatPath)
            notes.Add($"QAT path ON: fine-tune steps={TrainingConfig.QatFineTuneSteps}.");

        if (TrainingConfig.MixedPrecision && TrainingConfig.ForceCpu)
            notes.Add("Mixed precision ON with Force CPU (no CUDA AMP gain).");

        return string.Join(" ", notes);
    }

    private string BuildTrainingWizardDetailedText()
    {
        var lines = new List<string>
        {
            "Guided Wizards:",
            $"1) Multi-GPU setup: {(TrainingConfig.DistributedTraining && !string.Equals(TrainingConfig.MultiGpuStrategy, "none", StringComparison.OrdinalIgnoreCase) ? "OK" : "Configure Distributed + strategy")}.",
            $"2) Cluster profile: {(string.IsNullOrWhiteSpace(TrainingConfig.ClusterProfileName) ? "Select profile" : $"OK ({TrainingConfig.ClusterProfileName})")}.",
            $"3) RLHF import: {(TrainingConfig.FineTuneStageRlhf ? (string.IsNullOrWhiteSpace(TrainingConfig.RlhfFeedbackPath) && !string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase) ? "Set feedback path" : "OK") : "Optional")}.",
            $"4) Eval pack: {(string.IsNullOrWhiteSpace(TrainingConfig.EvalSuite) ? "Select eval suite" : $"OK ({TrainingConfig.EvalSuite})")}."
        };
        return string.Join("\n", lines);
    }

    private string? GetBlockingPreflightIssue()
    {
        if (TrainingConfig.FineTuningOrchestration
            && !TrainingConfig.FineTuneStageSft
            && !TrainingConfig.FineTuneStageDpo
            && !TrainingConfig.FineTuneStageRlhf)
            return "Fine-tuning orchestration is ON but no SFT/DPO/RLHF stage is selected.";
        if (TrainingConfig.FineTuneStageRlhf
            && !string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase)
            && string.IsNullOrWhiteSpace(TrainingConfig.RlhfFeedbackPath))
            return "RLHF stage requires feedback path when feedback source is not inline.";
        if (TrainingConfig.FineTuneStageRlhf
            && string.Equals(TrainingConfig.RlhfFeedbackSource, "inline", StringComparison.OrdinalIgnoreCase)
            && RlhfCollectedFeedback.Count == 0
            && string.IsNullOrWhiteSpace(TrainingConfig.RlhfFeedbackPath))
            return "RLHF inline mode requires collected feedback records. Add at least one record in Advanced Training.";

        if (TrainingConfig.OrchestratePipelineStages && !TrainingConfig.PipelineRunTrainStage)
            return "Pipeline orchestration is ON but Train stage is disabled. Enable train stage or disable orchestration.";

        if (!TrainingConfig.DistributedTraining) return null;

        var wsRaw = Environment.GetEnvironmentVariable("WORLD_SIZE");
        if (!int.TryParse(wsRaw, out var ws) || ws <= 1)
            return "Distributed training requires WORLD_SIZE > 1. Set WORLD_SIZE/RANK/LOCAL_RANK and retry, or disable Distributed mode.";
        if (TrainingConfig.ClusterWorldSize > 1 && ws != TrainingConfig.ClusterWorldSize)
            return $"Distributed preflight mismatch: WORLD_SIZE={ws} but ClusterWorldSize={TrainingConfig.ClusterWorldSize}. Align values before training.";
        if (string.Equals(TrainingConfig.MultiGpuStrategy, "none", StringComparison.OrdinalIgnoreCase))
            return "Distributed mode is ON but Multi-GPU strategy is 'none'. Select 'ddp' or 'fsdp', or disable Distributed mode.";

        return null;
    }

    private static string ResolveProjectRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var sln = Path.Combine(dir.FullName, "LLMForgeStudio.sln");
            var backendDir = Path.Combine(dir.FullName, "backends", "python");
            if (File.Exists(sln) && Directory.Exists(backendDir))
                return dir.FullName;
            dir = dir.Parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private void GoToWizardTarget()
    {
        if (!_wizardSetupDone) { SelectedSection = "Hardware"; return; }
        if (!_wizardDatasetImported) { SelectedSection = "Dataset"; return; }
        if (!_wizardTokenizerTrained || !_wizardPreviewBuilt) { SelectedSection = "Tokenization"; return; }
        if (!_wizardTrainingStarted) { SelectedSection = "Training"; return; }
        SelectedSection = "Generation";
    }

    private static string RunCommand(string fileName, string arguments)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            using var process = Process.Start(psi);
            if (process is null) return string.Empty;
            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(2500);
            return output;
        }
        catch
        {
            return string.Empty;
        }
    }
}
