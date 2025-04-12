using System.Text.Json;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddSingleton<ChatbotService>();

// Add CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:8000")
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();

app.UseHttpsRedirection();

// Chatbot endpoints
app.MapPost("/chat", async ([FromBody] ChatRequest request, ChatbotService chatbot) =>
{
    try
    {
        var response = await chatbot.GetResponse(request.Message);
        return Results.Ok(new ChatResponse(response));
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error in chat endpoint: {ex.Message}");
        return Results.BadRequest(new { error = ex.Message });
    }
})
.WithName("Chat")
.WithOpenApi();

app.Run();

// Models
record ChatRequest(string Message);
record ChatResponse(string Response);

// Chatbot Service
public class ChatbotService
{
    private readonly string _pythonScriptPath;

    public ChatbotService()
    {
        // Python'un tam yolunu bul
        var pythonPath = FindPythonPath();
        if (string.IsNullOrEmpty(pythonPath))
        {
            throw new Exception("Python not found. Please make sure Python is installed and in your PATH.");
        }

        // Python script'inin tam yolunu oluştur
        _pythonScriptPath = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "..", "chatbot_api.py"));
        if (!File.Exists(_pythonScriptPath))
        {
            throw new FileNotFoundException($"Python script not found at: {_pythonScriptPath}");
        }

        Console.WriteLine($"Using Python at: {pythonPath}");
        Console.WriteLine($"Using script at: {_pythonScriptPath}");
    }

    private string FindPythonPath()
    {
        var python3 = RunCommand("which python3");
        if (!string.IsNullOrEmpty(python3))
        {
            return python3.Trim();
        }

        var python = RunCommand("which python");
        if (!string.IsNullOrEmpty(python))
        {
            return python.Trim();
        }

        return null;
    }

    private string RunCommand(string command)
    {
        try
        {
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "/bin/bash",
                Arguments = $"-c \"{command}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (var process = System.Diagnostics.Process.Start(startInfo))
            {
                if (process == null)
                {
                    return null;
                }

                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                return output;
            }
        }
        catch
        {
            return null;
        }
    }

    public async Task<string> GetResponse(string message)
    {
        if (string.IsNullOrEmpty(message))
        {
            return "Please provide a message";
        }

        try
        {
            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = FindPythonPath(),
                Arguments = _pythonScriptPath,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using (var process = System.Diagnostics.Process.Start(startInfo))
            {
                if (process == null)
                {
                    throw new Exception("Failed to start Python process");
                }

                // Send message to Python script
                await process.StandardInput.WriteLineAsync(message);
                process.StandardInput.Close();

                // Read error output
                var errorOutput = await process.StandardError.ReadToEndAsync();
                if (!string.IsNullOrEmpty(errorOutput))
                {
                    Console.WriteLine("Python Output:");
                    Console.WriteLine(errorOutput);
                }

                // Read response from Python script
                var response = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (response.StartsWith("Error:"))
                {
                    throw new Exception(response);
                }

                return response.Trim();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error getting response: {ex.Message}");
            return "Sorry, I encountered an error while processing your message";
        }
    }
}

// Model data class
public class ModelData
{
    public List<string> vocabulary { get; set; }
    public List<string> intents { get; set; }
    public Dictionary<string, string[]> intents_responses { get; set; }
    public int input_size { get; set; }
    public int output_size { get; set; }
}