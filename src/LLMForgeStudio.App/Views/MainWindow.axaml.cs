using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using LLMForgeStudio.App.ViewModels;

namespace LLMForgeStudio.App.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
    }

    private async void OnImportDatasetClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || StorageProvider is null) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Import dataset",
            FileTypeFilter =
            [
                new FilePickerFileType("Text Files") { Patterns = ["*.txt", "*.md", "*.csv"] }
            ]
        });

        var file = files.FirstOrDefault();
        if (file is null) return;
        await vm.ImportDatasetFromPathAsync(file.Path.LocalPath);
    }

    private async void OnImportDatasetFolderClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || StorageProvider is null) return;

        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Import dataset folder"
        });

        var folder = folders.FirstOrDefault();
        if (folder is null) return;
        await vm.ImportDatasetFromFolderAsync(folder.Path.LocalPath);
    }

    private async void OnSaveProjectClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || StorageProvider is null) return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save project",
            SuggestedFileName = ".llmforge.json",
            FileTypeChoices = [new FilePickerFileType("LLM Forge Project") { Patterns = ["*.json"] }]
        });

        if (file is null) return;
        await vm.SaveProjectAsync(file.Path.LocalPath);
    }

    private async void OnLoadProjectClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || StorageProvider is null) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Load project",
            FileTypeFilter = [new FilePickerFileType("LLM Forge Project") { Patterns = ["*.json"] }]
        });

        var file = files.FirstOrDefault();
        if (file is null) return;
        await vm.LoadProjectAsync(file.Path.LocalPath);
    }

    private void OnOpenOutputClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || string.IsNullOrWhiteSpace(vm.RunDirectory)) return;
        var outputDir = Path.GetFullPath(vm.RunDirectory);
        Directory.CreateDirectory(outputDir);

        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = outputDir,
                UseShellExecute = true
            });
        }
        catch
        {
            vm.Log = $"Impossibile aprire cartella output: {outputDir}";
        }
    }

    private void OnOpenLlmFromScratchGuideClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        OpenUrl("https://github.com/angelos-p/llm-from-scratch/blob/main/docs/01-tokenization.md");
    }

    private void OnOpenAuthorWebsiteClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        OpenUrl("https://www.giovannidemiccoli.it");
    }

    private void OpenUrl(string url)
    {
        if (DataContext is not MainWindowViewModel vm) return;
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = url,
                UseShellExecute = true
            });
        }
        catch
        {
            vm.Log = $"Unable to open URL: {url}";
        }
    }
}
