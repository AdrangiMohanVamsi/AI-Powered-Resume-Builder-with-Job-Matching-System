import streamlit as st
from io import BytesIO
import google.generativeai as genai
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.flowables import HRFlowable  # <-- added
from reportlab.lib.enums import TA_CENTER

# Configure Gemini API for suggestions
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.warning("GOOGLE_API_KEY not found. Gemini features will be disabled.")
else:
    genai.configure(api_key=api_key)

# Initialize session state for dynamic entries
if 'work_experiences' not in st.session_state:
    st.session_state.work_experiences = [{"id": 1, 'job_title': '', 'company': '', 'start_date': '', 'end_date': '', 'responsibilities': ''}]
if 'education_entries' not in st.session_state:
    st.session_state.education_entries = [{"id": 1, 'degree': '', 'school': '', 'graduation_date': ''}]
if 'project_entries' not in st.session_state:
    st.session_state.project_entries = [{"id": 1, 'name': '', 'start_date': '', 'end_date': '', 'description': '', 'technologies': ''}]

st.title("Dynamic Resume Builder")

# --- Input Form ---
st.header("Personal Information")
name = st.text_input("Full Name")
email = st.text_input("Email")
phone = st.text_input("Phone Number")
linkedin = st.text_input("LinkedIn Profile URL")

st.header("Work Experience")
for i, exp in enumerate(st.session_state.work_experiences):
    st.subheader(f"Experience #{i + 1}")
    exp['job_title'] = st.text_input("Job Title", value=exp.get('job_title', ''), key=f"job_title_{exp['id']}")
    exp['company'] = st.text_input("Company", value=exp.get('company', ''), key=f"company_{exp['id']}")
    exp['start_date'] = st.text_input("Start Date (e.g., YYYY-MM)", value=exp.get('start_date', ''), key=f"start_date_{exp['id']}")
    exp['end_date'] = st.text_input("End Date (e.g., YYYY-MM or Present)", value=exp.get('end_date', ''), key=f"end_date_{exp['id']}")
    exp['responsibilities'] = st.text_area("Responsibilities (one per line)", value=exp.get('responsibilities', ''), key=f"responsibilities_{exp['id']}")

    if api_key:
        if st.button("Get AI Suggestions", key=f"suggest_responsibilities_{exp['id']}"):
            if exp['job_title']:
                with st.spinner(f"Generating suggestions for {exp['job_title']}..."):
                    try:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        prompt = f'''
                        You are a professional resume writer. 
                        For the job title "{exp['job_title']}", generate a list of 3-5 bullet points for responsibilities and achievements.
                        Focus on quantifiable achievements and use strong action verbs.
                        '''
                        response = model.generate_content(prompt)
                        
                        # Append suggestions to existing responsibilities
                        current_responsibilities = exp.get('responsibilities', '')
                        new_suggestions = response.text
                        exp['responsibilities'] = f"{current_responsibilities}\n{new_suggestions}".strip()
                        st.success("AI suggestions have been added to the responsibilities text area!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a Job Title first.")

if st.button("Add Another Experience"):
    new_id = max(e['id'] for e in st.session_state.work_experiences) + 1 if st.session_state.work_experiences else 1
    st.session_state.work_experiences.append({'id': new_id, 'job_title': '', 'company': '', 'start_date': '', 'end_date': '', 'responsibilities': ''})
    st.rerun()

st.header("Education")
for i, edu in enumerate(st.session_state.education_entries):
    st.subheader(f"Education #{i + 1}")
    edu['degree'] = st.text_input("Degree (e.g., B.S. in Computer Science)", value=edu.get('degree', ''), key=f"degree_{edu['id']}")
    edu['school'] = st.text_input("School", value=edu.get('school', ''), key=f"school_{edu['id']}")
    edu['graduation_date'] = st.text_input("Graduation Date (e.g., YYYY-MM)", value=edu.get('graduation_date', ''), key=f"graduation_date_{edu['id']}")

if st.button("Add Another Education"):
    new_id = max(e['id'] for e in st.session_state.education_entries) + 1 if st.session_state.education_entries else 1
    st.session_state.education_entries.append({'id': new_id, 'degree': '', 'school': '', 'graduation_date': ''})
    st.rerun()

st.header("Projects")
for i, proj in enumerate(st.session_state.project_entries):
    st.subheader(f"Project #{i + 1}")
    proj['name'] = st.text_input("Project Name", value=proj.get('name', ''), key=f"project_name_{proj['id']}")
    proj['description'] = st.text_area("Project Description (one bullet per line)", value=proj.get('description', ''), key=f"project_description_{proj['id']}")
    proj['technologies'] = st.text_input("Technologies Used (comma-separated)", value=proj.get('technologies', ''), key=f"project_technologies_{proj['id']}")

if st.button("Add Another Project"):
    new_id = max(p['id'] for p in st.session_state.project_entries) + 1 if st.session_state.project_entries else 1
    st.session_state.project_entries.append({'id': new_id, 'name': '', 'description': '', 'technologies': ''})
    st.rerun()

st.header("Skills")
skills = st.text_area("Skills (comma-separated)")

# --- Resume Generation with ReportLab ---
if st.button("Generate Resume"):
    if not name or not email or not phone:
        st.warning("Please fill in at least your name, email, and phone number.")
    else:
        with st.spinner("Generating PDF resume..."):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='NameStyle', fontName='Times-Roman', fontSize=24, alignment=1, spaceAfter=20))
            styles.add(ParagraphStyle(name='ContactStyle', fontName='Times-Roman', fontSize=10, alignment=1, spaceAfter=10))
            styles.add(ParagraphStyle(name='SectionHead', fontName='Times-Roman', fontSize=12, spaceBefore=12, spaceAfter=6, alignment=TA_CENTER))
            styles.add(ParagraphStyle(name='JobTitle', fontName='Times-Bold', fontSize=11, spaceAfter=2))
            styles.add(ParagraphStyle(name='JobDate', fontName='Times-Roman', fontSize=9, alignment=2))
            styles.add(ParagraphStyle(name='ResumeBullet', fontName='Times-Roman', fontSize=10, leftIndent=18, bulletIndent=0, spaceAfter=4, leading=12))
            styles.add(ParagraphStyle(name='SkillsStyle', fontName='Times-Roman', fontSize=10, leading=14))

            # <-- added: separator definition
            separator = HRFlowable(width="100%", thickness=0.7, lineCap='round', color=colors.black, spaceBefore=4, spaceAfter=7)
            header_separator = HRFlowable(width="100%",thickness=0.7,lineCap='round',color=colors.black,spaceBefore=1,spaceAfter=3)

            flowables = []

            # --- Build Document ---
            flowables.append(Paragraph(name, styles['NameStyle']))
            contact_info = f"{email} | {phone} | {linkedin}"
            flowables.append(Paragraph(contact_info, styles['ContactStyle']))

            # <-- added: line between header and Education
            flowables.append(separator)

            # Education
            flowables.append(Paragraph("Education".upper(), styles['SectionHead']))
            for edu in st.session_state.education_entries:
                if edu.get('degree') and edu.get('school'):
                    edu_text = f"<b>{edu['degree']}</b>, {edu['school']} - <i>{edu['graduation_date']}</i>"
                    flowables.append(Paragraph(edu_text, styles['ResumeBullet']))
            flowables.append(Spacer(1, 12))

            # <-- added: line between Education and Work Experience
            flowables.append(separator)

            # Work Experience
            flowables.append(Paragraph("Work Experience".upper(), styles['SectionHead']))
            for exp in st.session_state.work_experiences:
                if exp.get('job_title') and exp.get('company'):
                    # Using a table for side-by-side job title and date
                    header_data = [[Paragraph(f"{exp['job_title']}, {exp['company']}", styles['JobTitle']), 
                                    Paragraph(f"{exp['start_date']} - {exp['end_date']}", styles['JobDate'])]]
                    header_table = Table(header_data, colWidths=[doc.width - 1.5*inch, 1.5*inch])
                    flowables.append(header_table)
                    flowables.append(Spacer(1, 4))
                    for resp in exp['responsibilities'].splitlines():
                        if resp.strip():
                            flowables.append(Paragraph(resp.strip(), styles['ResumeBullet']))
                    flowables.append(Spacer(1, 10))

            # <-- added: line between Work Experience and Projects
            flowables.append(separator)

            # Projects
            flowables.append(Paragraph("Projects".upper(), styles['SectionHead']))
            for proj in st.session_state.project_entries:
                if proj.get('name'):
                    flowables.append(Paragraph(proj['name'], styles['JobTitle']))
                    flowables.append(Spacer(1, 4))
                    if proj.get('technologies'):
                        flowables.append(Paragraph(f"<i>Technologies: {proj['technologies']}</i>", styles['ResumeBullet']))
                    for desc in proj['description'].splitlines():
                        if desc.strip():
                            flowables.append(Paragraph(desc.strip(), styles['ResumeBullet']))
                    flowables.append(Spacer(1, 10))

            # <-- added: line between Projects and Skills
            flowables.append(separator)

            # Skills
            flowables.append(Paragraph("Skills".upper(), styles['SectionHead']))
            flowables.append(Paragraph(skills, styles['SkillsStyle']))

            doc.build(flowables)
            
            buffer.seek(0)
            
            st.download_button(
                label="Download Resume as PDF",
                data=buffer,
                file_name=f"{name.replace(' ', '_')}_Resume.pdf",
                mime="application/pdf"
            )
            st.success("Resume PDF generated successfully!")
