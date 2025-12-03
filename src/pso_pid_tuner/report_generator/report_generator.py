# ──────────────────────────────────────────────────────────────────────────────
# Project:       PID Optimizer
# Script:        report_generator.py
# Description:   Generates a complete optimization report including step and Bode plots,
#                PID results, simulation parameters, and plant data. Exports all figures and
#                summary information to a timestamped directory and builds a structured PDF
#                report using ReportLab.
#
# Authors:       Florin Büchi, Thomas Stähli
# Created:       01.12.2025
# Modified:      01.12.2025
# Version:       1.0
#
# License:       ZHAW Zürcher Hochschule für angewandte Wissenschaften (or internal use only)
# ──────────────────────────────────────────────────────────────────────────────


import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, PageTemplate, Frame)

from ..controlsys import bode_plot, crossover_frequency, PerformanceIndex, AntiWindup


def resource_path() -> Path:
    """Gibt den richtigen Pfad zur Ressource, auch nach PyInstaller-Build."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    else:
        # Script-Modus → zurück zur Projekt-Wurzel gehen
        project_root = Path(__file__).resolve().parent.parent
        return project_root


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
    performance_index: PerformanceIndex = data["performance_index"]
    best_performance_index = data["best_performance_index"]

    plant = data["plant"]
    pid = data["pid"]

    start_time = data["start_time"]
    end_time = data["end_time"]
    time_step = data["time_step"]
    sim_mode = data["sim_mode"]
    excitation_target = data["excitation_target"]

    plant_num = data["plant_num"]
    plant_den = data["plant_den"]

    anti_windup_method: AntiWindup = data["anti_windup_method"]
    constraint_min = data["constraint_min"]
    constraint_max = data["constraint_max"]

    # --------------------------
    # Print results
    # --------------------------
    n_char = 10
    print(f"""
    ✔ Optimization completed!

    swarm result:
       {'best_Kp':<{n_char}}= {best_Kp:8.2f}
       {'best_Ti':<{n_char}}= {best_Ti:8.2f}
       {'best_Td':<{n_char}}= {best_Td:8.2f}
       {"best_" + performance_index.name:<{n_char}}= {best_performance_index:8.4f}
    """)

    # Set parameters
    pid.set_pid_param(Kp=best_Kp, Ti=best_Ti, Td=best_Td)

    # --------------------------
    # Crossover frequency / Tf range
    # --------------------------
    L = lambda s: pid.controller(s) * plant.system(s)
    wc = crossover_frequency(L)

    # limitations of timeconstant of filter
    Tf_max = 1 / (100 * wc)  # can't be bigger, or filter would be too slow and impact the stepresponse
    fs_min = 10 / (np.pi * Tf_max)  # required minimum for sample frequency
    pid.set_filter(Tf=Tf_max)

    print("Recommended maximum value for the filter time constant Tf:")
    print(f"  Tf_max = {Tf_max:.4e} s   (limit imposed by the crossover frequency)")
    print(f"  For the generated plots, the filter time constant was set to Tf_max.")
    print(f"  Make sure to sample with at least {fs_min:.0f} Hz, otherwise the derivative filter will operate too close to the Nyquist limit.\n\n")

    # step response plant without feedback
    t_ol, y_ol = plant.step_response(
        t0=start_time,
        t1=end_time,
        dt=time_step)

    # build bode
    systems_for_bode = {}

    # Plot Step Response
    plt.figure(1)

    # step response feedbackloop
    match excitation_target:
        case "reference":
            t_cl, y_cl = pid.step_response(t0=start_time, t1=end_time, dt=time_step)
            systems_for_bode["Plant"] = plant.system
            systems_for_bode["Closed Loop"] = pid.closed_loop
            plt.plot(t_ol, y_ol, label="Plant")
            plt.plot(t_cl, y_cl, label="Closed Loop")
        case "input_disturbance":
            t_cl, y_cl = pid.step_response_l(t0=start_time, t1=end_time, dt=time_step)
            systems_for_bode["Closed Loop input disturbance"] = pid.closed_loop_l
            plt.plot(t_cl, y_cl, label="Closed Loop input disturbance")
        case "measurement_disturbance":
            t_cl, y_cl = pid.step_response_n(t0=start_time, t1=end_time, dt=time_step)
            systems_for_bode["Closed Loop measurement disturbance"] = pid.closed_loop_n
            plt.plot(t_cl, y_cl, label="Closed Loop measurement disturbance")

    plt.xlabel("time / s")
    plt.ylabel("output")
    plt.title("Step Response", fontweight="bold", fontname="Arial", pad=20)
    plt.grid(True)
    plt.legend()

    # Plot Bode
    bode_fig = bode_plot(systems_for_bode, high_exp = 5)

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
    logo_path = os.path.join(resource_path(), os.path.join("report_generator", "ZHAW_logo.png"))
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
    #elements.append(Spacer(1, 0.1 * cm))

    # INFO BLOCK
    elements.append(Paragraph("Information", style_h2))
    elements.append(Paragraph(
        '<font color="red"><b>PRELIMINARY – results have not been fully validated and the application is still under development'
        '</b></font>', style_body))
    elements.append(Paragraph("developed by: Thomas Staehli, Florin Büchi, Roland Büchi", style_body))
    elements.append(
        Paragraph("enjoy tuning and leave us some feedback: bhir@zhaw.ch", style_body))
    #elements.append(Spacer(1, 0.1 * cm))

    # STRUCTURE
    if anti_windup_method == AntiWindup.CLAMPING:
        aw_img_path = os.path.join(resource_path(), "report_generator", "reglerstruktur_clamping.png")
        aw_img = Image(aw_img_path, width=15 * cm, height=4.3 * cm)
    elif anti_windup_method == AntiWindup.CONDITIONAL:
        aw_img_path = os.path.join(resource_path(), "report_generator", "reglerstruktur_conditional.png")
        aw_img = Image(aw_img_path, width=15 * cm, height=5.3 * cm)

    aw_img.hAlign = "CENTER"
    elements.append(aw_img)
    #elements.append(Spacer(1, 0.5 * cm))

    # PLANT MODEL
    elements.append(Paragraph("Plant Model", style_h2))
    elements.append(Paragraph(f"Numerator: {plant_num}", style_body))
    elements.append(Paragraph(f"Denominator: {plant_den}", style_body))
    #elements.append(Spacer(1, 0.3 * cm))

    # PARAMETERS
    elements.append(Paragraph("Best Found Parameters", style_h2))
    elements.append(Paragraph(f"K<sub>p</sub> = {best_Kp:.4f}", style_body))
    elements.append(Paragraph(f"T<sub>i</sub> = {best_Ti:.4f}", style_body))
    elements.append(Paragraph(f"T<sub>d</sub> = {best_Td:.4f}", style_body))
    elements.append(Paragraph(f"Best {performance_index.name} = {best_performance_index:.6f}", style_body))
    #elements.append(Spacer(1, 0.3 * cm))

    # FILTER
    elements.append(Paragraph("Recommended Filter Time Constant (T<sub>f</sub>)", style_h2))
    elements.append(Paragraph(f"T<sub>f,max</sub> = {Tf_max:.4e} s   "f"(upper limit imposed by the crossover frequency)",style_body))
    elements.append(Paragraph(f"To implement this filter digitally, the sampling frequency "f"f<sub>s</sub> should be at least {fs_min:.0f} Hz.",style_body))
    elements.append(Paragraph(f"For the generated plots, the filter time constant was set to T<sub>f,max</sub>.",style_body))
    #elements.append(Spacer(1, 0.3 * cm))

    # SIM SETTINGS (2-column table)
    elements.append(Paragraph("Simulation Settings", style_h2))

    sim_table_data = [
        [Paragraph(f"Start time: {start_time}", style_body),
        Paragraph(f"End time: {end_time}", style_body)],

        [Paragraph(f"Mode: {sim_mode}", style_body),
        Paragraph(f"Time step: {time_step}", style_body)],

        [Paragraph(f"Excitation target: {excitation_target}", style_body),
        Paragraph(f"Anti-windup-method: {anti_windup_method.name.lower()}", style_body)],

        [Paragraph(f"Control output upper limit: {constraint_max}", style_body),
        Paragraph(f"Control output lower limit: {constraint_min}", style_body)]]

    sim_table = Table(sim_table_data, colWidths=[8 * cm, 8 * cm])
    sim_table.hAlign = "LEFT"
    sim_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    elements.append(sim_table)
    #elements.append(Spacer(1, 0.5 * cm))

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
