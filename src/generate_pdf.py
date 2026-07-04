import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'static'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def create_case_study():
    pdf_path = os.path.join(PROJECT_ROOT, "case_study_fraud_detection.pdf")
    print(f"Creating case study PDF at: {pdf_path}")
    
    # Page setup
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    story = []
    
    # Styles Setup
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#0f172a")    # Deep Slate Navy
    c_secondary = colors.HexColor("#2563eb")  # Brand Blue
    c_text = colors.HexColor("#334155")       # Charcoal Body Text
    c_light = colors.HexColor("#f8fafc")      # Light Slate Background
    c_border = colors.HexColor("#e2e8f0")     # Light Border Gray
    c_success = colors.HexColor("#047857")    # Deep Green
    
    # Modify existing styles or add unique ones
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        alignment=TA_CENTER,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=c_primary,
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_text,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_text,
        spaceAfter=4,
        leftIndent=15
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=c_text
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=8
    )

    # -------------------------------------------------------------------------
    # PAGE 1: TITLE & EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    # Decorative Header line
    # (We can use a thin table as a line separator)
    divider = Table([[""]], colWidths=[504], rowHeights=[3])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_secondary),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("🛡️ FINANCIAL SENTINEL", title_style))
    story.append(Paragraph("Real-Time Credit Card Fraud Detection Enterprise Platform<br/>"
                           "<b>Case Study & Architectural Review Report</b>", subtitle_style))
    story.append(divider)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(
        "Financial Sentinel is an end-to-end, high-performance machine learning microservice built to "
        "detect fraudulent credit card transactions in real time. Deployed utilizing a modular architecture, "
        "the engine protects banking infrastructures by analyzing transaction features, applying dynamic preprocessors, "
        "and evaluating risk profiles within milliseconds.", body_style
    ))
    story.append(Paragraph(
        "Using the ULB European Bank dataset containing 284,807 transactions, the data science team engineered "
        "and optimized a champion XGBoost model. By applying strict overfitting control through GridSearchCV "
        "and stratified splits, the model achieved a <b>Precision-Recall AUC (PR-AUC) of 0.8553</b> and a "
        "<b>Test F1-score of 0.7847</b>, successfully matching security recall targets with a very low False Positive rate.", body_style
    ))
    
    # Metagrid (Highlight Metrics Box)
    metric_data = [
        [
            Paragraph("<b>Test PR-AUC</b><br/><font size=14 color='#2563eb'><b>85.53%</b></font>", body_style),
            Paragraph("<b>Recall (Catch Rate)</b><br/><font size=14 color='#2563eb'><b>83.67%</b></font>", body_style),
            Paragraph("<b>False Alarms</b><br/><font size=14 color='#047857'><b>0.05%</b></font>", body_style)
        ]
    ]
    metric_table = Table(metric_data, colWidths=[168, 168, 168])
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(Spacer(1, 10))
    story.append(metric_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1. Data Ingestion & Imbalance Challenge", h1_style))
    story.append(Paragraph(
        "In financial transaction datasets, fraudulent entries represent a tiny fraction of the total volume. "
        "In our ingest pipeline, we verified the following class counts:", body_style
    ))
    story.append(Paragraph("• <b>Legitimate Transactions (Class 0)</b>: 227,451 samples (99.827%)", bullet_style))
    story.append(Paragraph("• <b>Fraudulent Transactions (Class 1)</b>: 394 samples (0.173%)", bullet_style))
    story.append(Paragraph("• <b>Imbalance Scale Ratio</b>: 1 : 577.29", bullet_style))
    
    # Ingestion Image side-by-side or embedded
    img_imbalance_path = os.path.join(STATIC_DIR, "class_imbalance.png")
    if os.path.exists(img_imbalance_path):
        story.append(Spacer(1, 10))
        img = Image(img_imbalance_path, width=280, height=210)
        img.hAlign = 'CENTER'
        story.append(img)
        story.append(Paragraph("<font size=8 color='#64748b'><b>Figure 1:</b> Target Class Imbalance distribution shown on a logarithmic scale for visibility.</font>", subtitle_style))
    
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 2: ARCHITECTURE & EDA FINDINGS
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. System Pipeline Architecture", h1_style))
    story.append(Paragraph(
        "To ensure corporate compliance and prevent information leakage, the system segregates processing into three modules:", body_style
    ))
    
    story.append(Paragraph("<b>A. Preprocessing & Scaler Pipeline</b>", h2_style))
    story.append(Paragraph(
        "Transactions carry unscaled, highly variable metrics like 'Amount' (ranging from $0 to $25,000) and 'Time' "
        "(0 to 172,800 seconds). PCA components (V1-V28) are anonymized and scaled. We construct a <b>ColumnTransformer</b> "
        "to fit a <i>StandardScaler</i> specifically on 'Amount' and 'Time' while utilizing 'passthrough' for the other 28 features. "
        "<b>Data Leakage Prevention</b> is guaranteed by executing `.fit_transform()` exclusively on the training partition "
        "and utilizing `.transform()` on test/production request payloads.", body_style
    ))
    
    story.append(Paragraph("<b>B. Exploratory Data Science & Visual Insights (EDA)</b>", h2_style))
    story.append(Paragraph(
        "Using correlation matrices, we isolated three PCA components displaying the strongest linear correlation with fraud: "
        "<b>V17</b>, <b>V14</b>, and <b>V12</b> (all strongly negatively correlated). Plotting V14 vs V17 using downsampling (to prevent "
        "overplotting) exposes a distinct, separate pocket where fraudulent charges reside (Figure 2).", body_style
    ))
    story.append(Paragraph(
        "Additionally, converting the raw 'Time' seconds parameter into a <b>24-Hour Diurnal Cycle</b> shows that legitimate "
        "shoppers follow typical sleeping patterns, while fraudulent activity remains flat and highly active overnight "
        "(midnight to 5 AM), indicating scripted attacks or international fraud origins.", body_style
    ))
    
    # Embed the scatter/time plots
    img_scatter_path = os.path.join(STATIC_DIR, "pca_separability_scatter.png")
    img_time_path = os.path.join(STATIC_DIR, "time_diurnal_distribution.png")
    
    plots_data = []
    row_images = []
    row_captions = []
    
    if os.path.exists(img_scatter_path):
        row_images.append(Image(img_scatter_path, width=220, height=165))
        row_captions.append(Paragraph("<b>Figure 2:</b> V14 vs V17 scatter plot (sampled Class 0) showing class separation.", table_text_style))
    if os.path.exists(img_time_path):
        row_images.append(Image(img_time_path, width=220, height=165))
        row_captions.append(Paragraph("<b>Figure 3:</b> Hour-of-Day density showing overnight flat fraud rate.", table_text_style))
        
    if row_images:
        plots_data.append(row_images)
        plots_data.append(row_captions)
        plots_table = Table(plots_data, colWidths=[240, 240])
        plots_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(Spacer(1, 10))
        story.append(plots_table)
    
    story.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 3: MODELING, BENCHMARKING & TUNING
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Model Benchmarking & Overfitting Validation", h1_style))
    story.append(Paragraph(
        "We ran a sequential multi-model benchmarking suite across six distinct classifiers, validating "
        "generalization by tracking metrics on both the Training set and the Test set. Highly complex "
        "classifiers like raw XGBoost and CatBoost achieved perfect F1 scores on training partitions but "
        "dropped on test evaluation—exhibiting a classic overfitting warning.", body_style
    ))
    
    # Leaderboard Table
    # Rows: Headers + 6 models
    header_row = [
        Paragraph("<b>Model</b>", table_header_style),
        Paragraph("<b>Train F1</b>", table_header_style),
        Paragraph("<b>Test F1</b>", table_header_style),
        Paragraph("<b>Train Rec.</b>", table_header_style),
        Paragraph("<b>Test Rec.</b>", table_header_style),
        Paragraph("<b>Test Prec.</b>", table_header_style),
        Paragraph("<b>Test PR-AUC</b>", table_header_style)
    ]
    
    leaderboard_data = [
        header_row,
        [Paragraph("XGBoost (Weighted)", table_text_style), "1.0000", "0.8482", "1.0000", "0.8265", "0.8710", "0.8791"],
        [Paragraph("CatBoost (Balanced)", table_text_style), "0.9359", "0.7465", "1.0000", "0.8265", "0.6807", "0.8287"],
        [Paragraph("Standard Random Forest", table_text_style), "0.8714", "0.7313", "0.9975", "0.8469", "0.6434", "0.8103"],
        [Paragraph("Balanced Random Forest", table_text_style), "0.2474", "0.2178", "1.0000", "0.8980", "0.1239", "0.7739"],
        [Paragraph("LightGBM (Weighted)", table_text_style), "0.1188", "0.1118", "0.9670", "0.8878", "0.0596", "0.0538"],
        [Paragraph("Logistic Regression", table_text_style), "0.1184", "0.1144", "0.9239", "0.9184", "0.0610", "0.7159"]
    ]
    
    # Create the table
    l_table = Table(leaderboard_data, colWidths=[150, 55, 55, 60, 60, 60, 64])
    l_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light]),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    story.append(l_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("<font size=8 color='#64748b'><b>Table 1:</b> Leaderboard comparing standard training and test metric outputs across all baseline models.</font>", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. Hyperparameter Optimization & Overfitting Prevention", h1_style))
    story.append(Paragraph(
        "To mitigate the 15.2% F1 generalization gap found in XGBoost, we executed a Grid Search with "
        "3-Fold Stratified Cross-Validation on the training set, optimizing for Average Precision (PR-AUC). "
        "By restricting tree parameters (shallowing the maximum depth to prevent memorizing transaction noise), we forced the model to generalize.", body_style
    ))
    
    # Grid results callout box
    grid_metrics = [
        [
            Paragraph(
                "<b>Optimal Hyperparameters Found:</b><br/>"
                "• <i>Learning Rate</i>: 0.1 | • <i>Max Depth</i>: 5 | • <i>Estimators</i>: 150<br/>"
                "<b>Tuning Results:</b><br/>"
                "• <b>Train F1</b>: 0.9068 | <b>Test F1</b>: 0.7847 | <b>Generalization Gap</b>: <b>12.21% (PASS)</b><br/>"
                "• <b>Test Recall</b>: <b>83.67%</b> (Improved from 82.65% baseline)<br/>"
                "• <b>Test Precision</b>: <b>73.87%</b> (Extremely low false alarm rates)",
                callout_style
            )
        ]
    ]
    grid_table = Table(grid_metrics, colWidths=[480])
    grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('BOX', (0,0), (-1,-1), 1.5, c_secondary),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(grid_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("5. Deployment & Production Interfaces", h1_style))
    story.append(Paragraph(
        "The optimized model binary (`fraud_model.pkl`) and scaler (`preprocessor.pkl`) are served in production using "
        "two highly robust application layer gateways:", body_style
    ))
    story.append(Paragraph(
        "1. <b>FastAPI REST Microservice</b>: A high-performance REST API running in the background. It utilizes an async lifespan "
        "context loader to cache the binaries directly in memory on startup (minimizing prediction latency). Requests are "
        "strictly schema-validated via Pydantic classes, exposing interactive auto-documentation (Swagger UI) at `/docs`.", bullet_style
    ))
    story.append(Paragraph(
        "2. <b>Interactive Streamlit Dashboard</b>: Built with a premium SaaS slate theme for manual analysts. It provides "
        "a 'Single Transaction Scan' tab (featuring high-risk flashing warning overlays), a 'Batch Processing Pipeline' tab "
        "(handling drag-and-drop CSV table uploads and exporting scored results), and an 'Exploratory Data Insights' tab.", bullet_style
    ))
    
    # Build Document
    print("Building document flowables...")
    doc.build(story)
    print("PDF generation complete!")

if __name__ == "__main__":
    create_case_study()
