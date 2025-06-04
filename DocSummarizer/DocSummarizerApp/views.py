from django.shortcuts import render
from django.http import JsonResponse
from .forms import DocumentForm
import pdfplumber
import pandas as pd
from .api import client


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    text = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')
    return text

def extract_text_from_txt(file):
    return file.read().decode('utf-8')

def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_text_with_groq(text):
    
    prompt = f"Summarize:\n\n{text}"
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # Lower temperature for more focused responses
        max_tokens=2024,   # Limit the number of tokens to control summary length
        top_p=1,
        stream=False
    )
    
    summary = completion.choices[0].message.content.strip() if completion.choices else ""
    return summary


def summarize_large_text_with_groq(text, chunk_size=1000):
    chunks = split_text(text, chunk_size)
    chunk_summaries = [summarize_text_with_groq(chunk) for chunk in chunks]
    combined_summary = " ".join(chunk_summaries)
    return summarize_text_with_groq(combined_summary)

def upload_file(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save()
            file = document.file
            text = ""
            
            if file.name.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.name.endswith('.xlsx'):
                text = extract_text_from_excel(file)
            elif file.name.endswith('.txt'):
                text = extract_text_from_txt(file)
            else:
                return JsonResponse({'error': 'Unsupported file type'}, status=400)
            
            # Summarize the text using Groq
            summary = summarize_large_text_with_groq(text)
            return JsonResponse({'summary': summary})
    else:
        form = DocumentForm()
    return render(request, 'upload.html', {'form': form})
