"""Generate a clean, high-visibility, presentation-ready PDF of the Video Presentation Script
for Yash Ranpariya (PRN 272251066).
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


def generate_script_pdf(output_pdf_path: Path):
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    header_title = ParagraphStyle(
        "HeaderTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0D3B66"),
        alignment=1,
    )

    header_sub = ParagraphStyle(
        "HeaderSub",
        fontName="Helvetica",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#596A7A"),
        alignment=1,
    )

    sec_title = ParagraphStyle(
        "SecTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0D3B66"),
        spaceBefore=10,
        spaceAfter=4,
    )

    cue_label = ParagraphStyle(
        "CueLabel",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#B75D00"),
    )

    cue_text = ParagraphStyle(
        "CueText",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2C3E50"),
    )

    speech_label = ParagraphStyle(
        "SpeechLabel",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1A5F7A"),
    )

    speech_text = ParagraphStyle(
        "SpeechText",
        fontName="Helvetica",
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor("#1D2D44"),
    )

    meta_text = ParagraphStyle(
        "MetaText",
        fontName="Helvetica",
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("CCA 3 — COMPUTER VISION PRACTICAL ASSIGNMENT", header_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Official 6–8 Minute Video Presentation & Spoken Dialogue Script", header_sub))
    story.append(Spacer(1, 8))

    # Student Details Bar
    meta_data = [
        [
            Paragraph("<b>Student Name:</b> Yash Ranpariya", meta_text),
            Paragraph("<b>PRN:</b> 272251066", meta_text),
            Paragraph("<b>Institution:</b> MIT-WPU Pune", meta_text),
        ],
        [
            Paragraph("<b>Course:</b> MCA (Term III)", meta_text),
            Paragraph("<b>Faculty:</b> Prof. Rahul Inchal", meta_text),
            Paragraph("<b>GitHub:</b> github.com/Yash-Ranpariya/cv-", meta_text),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[175, 140, 215])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F4F8")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    sections = [
        {
            "time": "[0:00 – 1:00] Section 1: Introduction & Dataset Files",
            "show": "Open <b>main.py</b> in VS Code and show the <b>dataset/</b> folder with original_photo.jpg, second_photo.jpg, and original_video.mp4.",
            "say": "Hello, respected faculty and evaluators. My name is <b>Yash Ranpariya</b>, PRN <b>272251066</b>, studying Master of Computer Applications at MIT World Peace University, Pune. "
                   "Today, I am presenting my Continuous Assessment 3 practical coding assignment in Computer Vision, covering Units 3, 4, and 5, under the guidance of Prof. Rahul Inchal.<br/><br/>"
                   "For this assignment, I captured my own authentic dataset without using stock photos: "
                   "First, <b>original_photo.jpg</b> for restoration, feature extraction, and recognition; "
                   "Second, <b>second_photo.jpg</b> from an alternate perspective for feature matching; and "
                   "Third, <b>original_video.mp4</b>, a self-recorded clip for motion estimation. "
                   "Let us now walk through the implementation and results for each task."
        },
        {
            "time": "[1:00 – 2:30] Section 2: Task 1 — Image Restoration & Segmentation (Unit 3)",
            "show": "Open <b>outputs/task1_restoration_segmentation/fig4_comprehensive_restoration_grid.png</b> and <b>fig8_watershed_segmentation_pipeline.png</b>.",
            "say": "Task 1 addresses two problems: image restoration and semantic object segmentation.<br/><br/>"
                   "<b>Part A: Image Restoration:</b> I degraded my photo with two noise models: Additive White Gaussian Noise (AWGN, sigma=25) simulating electronic sensor noise, "
                   "and Salt-and-Pepper impulse noise (3%) simulating transmission bit errors.<br/>"
                   "I applied both a linear 2D Gaussian filter and a non-linear Median filter. "
                   "On Gaussian noise, Gaussian filtering achieved <b>28.82 dB PSNR and 0.7405 SSIM</b>. "
                   "However, on Salt-and-Pepper noise, Gaussian filtering smeared impulses into cloudy patches (26.67 dB). "
                   "In contrast, the <b>Median filter achieved an outstanding 31.31 dB PSNR and 0.8949 SSIM</b> because sorting the neighborhood completely rejects extreme 0 and 255 outliers, keeping edges sharp.<br/><br/>"
                   "<b>Part B: Morphological Operations & Segmentation:</b> I applied Otsu's thresholding to maximize between-class variance, followed by morphological erosion, dilation, opening, and closing. "
                   "To segment touching regions, I computed the Euclidean Distance Transform to extract sure foreground seeds and executed marker-controlled <b>Watershed Segmentation</b>, highlighted by the red boundary lines."
        },
        {
            "time": "[2:30 – 4:00] Section 3: Task 2 — Feature Extraction & Invariant Matching (Unit 4)",
            "show": "Open <b>outputs/task2_feature_extraction/fig1_hog_visualization.png</b> and <b>fig3_matching_comparison_summary.png</b>.",
            "say": "Task 2 investigates local gradient and scale-invariant descriptors.<br/><br/>"
                   "<b>Histogram of Oriented Gradients (HOG):</b> HOG computes 9-bin gradient histograms across normalized 8x8 pixel cells. It extracted <b>489,168 features</b> from my photo, capturing prominent object contours visible in the inferno visualization.<br/><br/>"
                   "<b>Scale-Invariant Feature Transform (SIFT):</b> SIFT detects stable extrema in Difference-of-Gaussians scale space, assigns a dominant orientation, and builds a 128-dimensional descriptor. "
                   "I detected 2,000 keypoints per photo and matched them using Lowe's Ratio Test (0.75 cutoff) and RANSAC homography. "
                   "On the original scene pair, SIFT achieved <b>220 good matches with 111 RANSAC inliers (50.5% inlier ratio)</b>.<br/>"
                   "To test invariance, I rotated Photo 2 by <b>35 degrees and scaled it to 0.8x</b>. SIFT maintained <b>219 matches and 106 inliers (48.4% inlier ratio)</b>, experimentally proving scale and rotational invariance."
        },
        {
            "time": "[4:00 – 5:30] Section 4: Task 3 — Object Recognition (Viola–Jones vs YOLOv8) (Unit 5)",
            "show": "Open <b>outputs/task3_object_recognition/fig1_detection_comparison_montage.png</b>.",
            "say": "Task 3 contrasts classical geometric detection against deep convolutional neural networks.<br/><br/>"
                   "<b>Viola–Jones Haar Cascade:</b> Uses rectangular Haar features accelerated by an Integral Image and selected via AdaBoost in an attentional cascade. "
                   "It successfully detected <b>2 frontal faces</b> (blue boxes). However, it is restricted to frontal human faces with no continuous confidence score, failing when head tilt exceeds 20 degrees.<br/><br/>"
                   "<b>Ultralytics YOLOv8:</b> An anchor-free deep single-stage CNN with a CSPDarknet backbone and PANet pyramid. "
                   "It detected <b>3 objects</b> across MS COCO classes with calibrated continuous probability scores and high angle tolerance.<br/>"
                   "This highlights the evolution from rigid handcrafted heuristics to generalized hierarchical representation learning."
        },
        {
            "time": "[5:30 – 7:00] Section 5: Task 4 — Video Motion Estimation (Unit 5)",
            "show": "Open <b>outputs/task4_motion_estimation/fig1_motion_estimation_comparison_grid.png</b> and <b>08_lucas_kanade_motion_vectors.png</b>.",
            "say": "Task 4 analyzes temporal dynamics in my video using four complementary algorithms:<br/><br/>"
                   "1. <b>Temporal Frame Differencing:</b> Subtracted adjacent frames to detect <b>6 motion clusters</b>. Very fast, but subject to ghosting when motion pauses.<br/>"
                   "2. <b>GMM Background Subtraction (MOG2):</b> Models each background pixel with 3 to 5 Gaussians, isolating moving silhouettes while filtering dynamic shadows.<br/>"
                   "3. <b>Lucas–Kanade Sparse Optical Flow:</b> Tracks Shi-Tomasi corners under the brightness constancy assumption. It computed <b>215 motion vectors with an average speed of 2.64 px/frame</b>, rendered as yellow arrows indicating travel direction.<br/>"
                   "4. <b>Dense Farneback Flow:</b> Computes pixel-wise flow visualized via an HSV color wheel (hue represents direction, brightness represents velocity).<br/><br/>"
                   "While frame differencing and GMM provide spatial presence masks, optical flow delivers true velocity vectors essential for tracking."
        },
        {
            "time": "[7:00 – 7:30] Section 6: Conclusion & Repository Verification",
            "show": "Show terminal with <b>python main.py --validate</b> PASS status, or the GitHub repository at <b>github.com/Yash-Ranpariya/cv-</b>.",
            "say": "In conclusion, this project implemented and verified the entire computer vision continuum: restoration, segmentation, invariant feature extraction, object recognition, and multi-paradigm video motion estimation. "
                   "All source code is committed to my GitHub repository at <b>github.com/Yash-Ranpariya/cv-</b>, and the original video recording has been shared on Google Drive with Prof. Rahul Inchal.<br/><br/>"
                   "Thank you very much for your time and guidance."
        }
    ]

    for sec in sections:
        sec_flow = []
        sec_flow.append(Paragraph(sec["time"], sec_title))

        # Cue Box (What to show)
        cue_table = Table(
            [[Paragraph("<b>ON SCREEN:</b>", cue_label), Paragraph(sec["show"], cue_text)]],
            colWidths=[80, 452],
        )
        cue_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E7")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#F6AD55")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        sec_flow.append(cue_table)
        sec_flow.append(Spacer(1, 4))

        # Speech Box (What to say)
        speech_table = Table(
            [[Paragraph("<b>SPOKEN WORDS:</b>", speech_label), Paragraph(sec["say"], speech_text)]],
            colWidths=[80, 452],
        )
        speech_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F7FB")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#90CDF4")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        sec_flow.append(speech_table)
        sec_flow.append(Spacer(1, 8))

        story.append(KeepTogether(sec_flow))

    doc.build(story)
    return output_pdf_path


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "Yash_Ranpariya_Video_Script.pdf"
    generate_script_pdf(out_file)
    print(f"Generated Script PDF: {out_file}")
