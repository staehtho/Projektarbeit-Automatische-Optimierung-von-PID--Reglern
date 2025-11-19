import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, PageTemplate, Frame)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import sys, os

# TODO: Lösung finden
# Ensure project root (pso_pid_tuner) is in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from controlsys.utils import bode_plot, crossover_frequency

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def add_footer(canvas, doc):
    canvas.saveState()
    width, height = A4

    # footnote left
    canvas.setFont("Helvetica", 9)
    canvas.drawString(2 * cm, 1 * cm, "PID-Optimizer")

    # footnote right
    page_text = f"Page {doc.page}"
    canvas.drawRightString(width - 2 * cm, 1 * cm, page_text)

    canvas.restoreState()


def report_generator(data: dict):
    """
    Receives a dictionary with all required variables for plotting,
    exporting and PDF building.
    """

    # --------------------------
    # Extract data from dictionary
    # --------------------------
    best_Kp = data["best_Kp"]
    best_Ti = data["best_Ti"]
    best_Td = data["best_Td"]
    best_itae = data["best_itae"]

    plant = data["plant"]
    pid = data["pid"]

    start_time = data["start_time"]
    end_time = data["end_time"]
    time_step = data["time_step"]
    excitation_target = data["excitation_target"]

    plant_num = data["plant_num"]
    plant_den = data["plant_den"]

    # --------------------------
    # Print results
    # --------------------------
    print(f"""
    ✔ Optimization completed!

    swarm result:
       {'best_Kp':<10}= {best_Kp:8.2f}
       {'best_Ti':<10}= {best_Ti:8.2f}
       {'best_Td':<10}= {best_Td:8.2f}
       {'best_itae':<10}= {best_itae:8.4f}
    """)

    # Set parameters
    pid.set_pid_param(Kp=best_Kp, Ti=best_Ti, Td=best_Td)

    # --------------------------
    # Crossover frequency / Tf range
    # --------------------------
    L = lambda s: pid.controller(s) * plant.system(s)
    wc = crossover_frequency(L)
    fs = 20000 # Hz, TODO: später aus Plant übernehmen

    # limitations of timeconstant of filter
    Tf_max = 1 / (100 * wc)  # can't be bigger, or filter would be too slow and impact the stepresponse
    Tf_min = 10 / (np.pi * fs)  # can't be smaller, or filter would be too close to Nyquistfrequency
    pid.set_filter(Tf=Tf_max)

    print("Recommended range for the filter time constant Tf:")
    print(f"  Tf_min = {Tf_min:.6e} s   (limit imposed by the sampling frequency: {fs}Hz)")
    print(f"  Tf_max = {Tf_max:.6e} s   (limit imposed by the crossover frequency)")
    print(f"→ Choose Tf such that  {Tf_min:.6e}  ≤  Tf  ≤  {Tf_max:.6e}")
    print(f"  For the generated plots, the filter time constant was set to Tf_max\n\n")

    # step response plant without feedback
    t_ol, y_ol = plant.step_response(
        t0=start_time,
        t1=end_time,
        dt=time_step)

    # build bode
    systems_for_bode = {
        "Open Loop": plant.system
    }

    # step response feedbackloop
    match excitation_target:
        case "reference":
            t_cl, y_cl = pid.step_response(
                t0=start_time,
                t1=end_time,
                dt=time_step)
            systems_for_bode["Closed Loop Reference"] = pid.closed_loop
        case "input_disturbance":
            t_cl, y_cl = pid.z1_step_response(
                t0=start_time,
                t1=end_time,
                dt=time_step)
            systems_for_bode["Closed Loop Z1"] = pid.closed_loop_Z1
        case "measurement_disturbance":
            t_cl, y_cl = pid.z2_step_response(
                t0=start_time,
                t1=end_time,
                dt=time_step)
            systems_for_bode["Closed Loop Z2"] = pid.closed_loop_Z2
        case _:
            t_cl = np.zeros_like(t_ol)
            y_cl = np.zeros_like(t_ol)

        # Plot Step Response
    plt.figure(1)
    if excitation_target == "reference":
        plt.plot(t_ol, y_ol, label="Open Loop")
    plt.plot(t_cl, y_cl, label="Closed Loop")
    plt.xlabel("time / s")
    plt.ylabel("output")
    plt.title("Step Response", fontweight="bold", fontname="Arial", pad=20)
    plt.grid(True)
    plt.legend()

    # Plot Bode
    bode_fig = bode_plot(systems_for_bode)

    # Get user's download directory
    download_dir = os.path.join(os.path.expanduser("~"), "Downloads")

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create run-specific folder with timestamp and label
    run_output_dir = os.path.join(download_dir, f"PID_Optimization_Result_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # export bodeplots
    step_plot_path = os.path.join(run_output_dir, f"step_response_{timestamp}.png")
    bode_plot_path = os.path.join(run_output_dir, f"bode_{timestamp}.png")

    plt.figure(1)
    plt.savefig(step_plot_path, dpi=600)

    plt.figure(2)
    bode_fig.savefig(bode_plot_path, dpi=600)

    print(f"{'Step response saved to:':<32} {step_plot_path}")
    print(f"{'Bode plot saved to:':<32} {bode_plot_path}")

    # create pdf
    pdf_path = os.path.join(run_output_dir, f"results_{timestamp}.pdf")

    styles = getSampleStyleSheet()
    style_h1 = styles["Heading1"]
    style_h2 = styles["Heading2"]
    style_body = styles["BodyText"]

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)

    # Frame
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id='normal'
    )

    # PageTemplate with Footer
    doc.addPageTemplates([
        PageTemplate(id='AllPages', frames=frame, onPage=add_footer)
    ])

    elements = []

    # LOGO
    logo_path = os.path.join(BASE_DIR, "ZHAW_logo.png")
    logo = Image(logo_path, width=3 * cm, height=3 * cm)

    # Title + timestamp stacked in one cell
    title_paragraph = Paragraph("PID Optimization Results", style_h1)
    timestamp_paragraph = Paragraph(f"Generated on <b>{timestamp}</b>", style_body)

    # Combine title + timestamp vertically
    title_block = [title_paragraph, timestamp_paragraph]

    # 2-column table: left title block, right logo
    header_table = Table(
        [[title_block, logo]],
        colWidths=[13 * cm, 4 * cm]  # adjust if needed
    )

    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),

        ("TOPPADDING", (1, 0), (1, 0), -16),

        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 0.7 * cm))

    # INFO BLOCK
    elements.append(Paragraph("Information", style_h2))
    elements.append(Paragraph(
        '<font color="red"><b>PRELIMINARY – results have not been fully validated and the application is still under development'
        '</b></font>', style_body))
    elements.append(Paragraph("developed by: Thomas Staehli, Florin Büchi, Roland Büchi", style_body))
    elements.append(
        Paragraph("enjoy tuning and leave us some feedback: https://buymeacoffee.com/SwarmAndOrder", style_body))
    elements.append(Spacer(1, 1 * cm))

    # PARAMETERS
    elements.append(Paragraph("Best Found Parameters", style_h2))
    elements.append(Paragraph(f"K<sub>p</sub> = {best_Kp:.4f}", style_body))
    elements.append(Paragraph(f"T<sub>i</sub> = {best_Ti:.4f}", style_body))
    elements.append(Paragraph(f"T<sub>d</sub> = {best_Td:.4f}", style_body))
    elements.append(Paragraph(f"Best ITAE = {best_itae:.6f}", style_body))
    elements.append(Spacer(1, 0.5 * cm))

    # FILTER
    elements.append(Paragraph("Recommended Filter Time Constant (T<sub>f</sub>)", style_h2))
    elements.append(Paragraph(
        f"T<sub>f,min</sub> = {Tf_min:.6e} s   "f"(limit imposed by the sampling frequency: {fs} Hz)", style_body))
    elements.append(Paragraph(
        f"T<sub>f,max</sub> = {Tf_max:.6e} s   "f"(limit imposed by the crossover frequency)", style_body))
    elements.append(Paragraph(
        f"&#8594; Choose T<sub>f</sub> such that  "f"{Tf_min:.6e}   &#8804;   T<sub>f</sub>   &#8804;   {Tf_max:.6e}",
        style_body))
    elements.append(Paragraph(
        "For the generated plots, the filter time constant was set to T<sub>f,max</sub>.", style_body))
    elements.append(Spacer(1, 0.5 * cm))

    # SIM SETTINGS
    elements.append(Paragraph("Simulation Settings", style_h2))
    elements.append(Paragraph(f"Start time: {start_time}", style_body))
    elements.append(Paragraph(f"End time: {end_time}", style_body))
    elements.append(Paragraph(f"Time step: {time_step}", style_body))
    elements.append(Paragraph(f"Excitation target: {excitation_target}", style_body))
    elements.append(Spacer(1, 0.5 * cm))

    # PLANT MODEL
    elements.append(Paragraph("Plant Model", style_h2))
    elements.append(Paragraph(f"Numerator: {plant_num}", style_body))
    elements.append(Paragraph(f"Denominator: {plant_den}", style_body))
    elements.append(Spacer(1, 1 * cm))

    # PAGE BREAK
    elements.append(PageBreak())

    # PLOTS
    elements.append(Image(step_plot_path, width=16 * cm, height=11 * cm))
    elements.append(Spacer(1, 1 * cm))
    elements.append(Image(bode_plot_path, width=16 * cm, height=11 * cm))

    # Build
    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)

    print(f"{'PDF written to:':<32} {pdf_path}")
    os.startfile(run_output_dir)

    plt.show()
