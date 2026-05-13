namespace LLMForgeStudio.App.Core.Tokenization;

public sealed record VocabularyItem(int Id, string Token, int Frequency = 0);
