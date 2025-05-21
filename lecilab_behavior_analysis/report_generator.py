# generates a report of the behavior for the animals in the training village
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Create a PDF file to hold multiple figures
with PdfPages('output_figures.pdf') as pdf:
    for i in range(5):
        # Create figure
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        ax.plot(x, y)
        ax.set_title(f'Figure {i+1}: Sine wave with phase {i}')
        ax.set_xlabel('x')
        ax.set_ylabel('sin(x + phase)')
        
        # Save the current figure to the PDF
        pdf.savefig(fig)
        plt.close(fig)

    # Add a metadata page
    d = pdf.infodict()
    d['Title'] = 'Generated PDF with Figures'
    d['Author'] = 'Your Name'
    d['Subject'] = 'Test of PDF generation'
