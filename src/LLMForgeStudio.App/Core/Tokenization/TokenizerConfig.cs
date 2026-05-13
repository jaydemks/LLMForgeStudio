namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class TokenizerConfig
{
    public TokenizerKind Kind { get; set; } = TokenizerKind.Character;
    public bool LowercaseBeforeTraining { get; set; } = false;
    public int TargetVocabSize { get; set; } = 512;
    public int MinFrequency { get; set; } = 2;
    public int MaxMerges { get; set; } = 300;
    public bool KeepPunctuationAsTokens { get; set; } = true;
    public bool UseCharacterFallback { get; set; } = true;
    public string Notes { get; set; } = string.Empty;
}
