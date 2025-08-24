import streamlit as st
from io import BytesIO
import os
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Configure Gemini API for summary generation
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Initialize session state
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
    exp['start_date'] = st.text_input("Start Date (MM YYYY)", value=exp.get('start_date', ''), key=f"start_date_{exp['id']}")
    exp['end_date'] = st.text_input("End Date (MM YYYY or Present)", value=exp.get('end_date', ''), key=f"end_date_{exp['id']}")
    exp['responsibilities'] = st.text_area("Responsibilities (one per line)", value=exp.get('responsibilities', ''), key=f"responsibilities_{exp['id']}")

if st.button("Add Another Experience"):
    new_id = max(e['id'] for e in st.session_state.work_experiences) + 1
    st.session_state.work_experiences.append({'id': new_id, 'job_title': '', 'company': '', 'start_date': '', 'end_date': '', 'responsibilities': ''})
    st.rerun()

st.header("Education")
for i, edu in enumerate(st.session_state.education_entries):
    st.subheader(f"Education #{i + 1}")
    edu['degree'] = st.text_input("Degree", value=edu.get('degree', ''), key=f"degree_{edu['id']}")
    edu['school'] = st.text_input("School", value=edu.get('school', ''), key=f"school_{edu['id']}")
    edu['graduation_date'] = st.text_input("Duration (YYYY - YYYY)", value=edu.get('graduation_date', ''), key=f"graduation_date_{edu['id']}")

if st.button("Add Another Education"):
    new_id = max(e['id'] for e in st.session_state.education_entries) + 1
    st.session_state.education_entries.append({'id': new_id, 'degree': '', 'school': '', 'graduation_date': ''})
    st.rerun()

st.header("Projects")
for i, proj in enumerate(st.session_state.project_entries):
    st.subheader(f"Project #{i + 1}")
    proj['name'] = st.text_input("Project Name", value=proj.get('name', ''), key=f"project_name_{proj['id']}")
    proj['description'] = st.text_area("Project Description (one bullet per line)", value=proj.get('description', ''), key=f"project_description_{proj['id']}")
    proj['technologies'] = st.text_input("Technologies Used", value=proj.get('technologies', ''), key=f"project_technologies_{proj['id']}")

if st.button("Add Another Project"):
    new_id = max(p['id'] for p in st.session_state.project_entries) + 1
    st.session_state.project_entries.append({'id': new_id, 'name': '', 'description': '', 'technologies': ''})
    st.rerun()

st.header("Skills")
skills = st.text_area("Skills (comma-separated)")

# --- Resume Generation ---
if st.button("Generate Resume"):
    if not name or not email or not phone:
        st.warning("Please fill in at least your name, email, and phone number.")
    else:
        # Generate professional summary using Gemini
        professional_summary = ""
        if api_key:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                user_data = f"""
                **Personal Information:**
                - Name: {name}
                - Email: {email}
                - Phone: {phone}
                - LinkedIn: {linkedin}

                **Work Experience:**
                """
                for exp in st.session_state.work_experiences:
                    user_data += f"""
                    - Job Title: {exp['job_title']}
                      Company: {exp['company']}
                      Duration: {exp['start_date']} to {exp['end_date']}
                      Responsibilities: {exp['responsibilities']}
                    """
                user_data += """
                **Education:**
                """
                for edu in st.session_state.education_entries:
                    user_data += f"""
                    - Degree: {edu['degree']}
                      School: {edu['school']}
                      Graduation Date: {edu['graduation_date']}
                    """
                user_data += """
                **Projects:**
                """
                for proj in st.session_state.project_entries:
                    user_data += f"""
                    - Project Name: {proj['name']}
                      Description: {proj['description']}
                      Technologies: {proj['technologies']}
                    """
                user_data += f"""
                **Skills:**
                {skills}
                """
                prompt = f"""
                Based on the following resume details, write a compelling and professional summary of 2-3 sentences.
                Highlight the candidate's key skills, experience, and achievements.
                Make it concise and impactful.

                {user_data}
                """
                response = model.generate_content(prompt)
                professional_summary = response.text
            except Exception as e:
                st.error(f"Summary generation failed: {e}")

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='NameStyle', fontName='Times-Bold', fontSize=20, alignment=TA_CENTER, spaceAfter=14))
        styles.add(ParagraphStyle(name='ContactStyle', fontName='Times-Roman', fontSize=10, alignment=TA_CENTER, spaceAfter=14))
        styles.add(ParagraphStyle(name='SectionHead', fontName='Times-Bold', fontSize=12, alignment=TA_LEFT, spaceBefore=14, spaceAfter=6))
        styles.add(ParagraphStyle(name='JobTitle', fontName='Times-Bold', fontSize=11, spaceAfter=2))
        styles.add(ParagraphStyle(name='JobDate', fontName='Times-Roman', fontSize=9, alignment=TA_RIGHT))
        styles.add(ParagraphStyle(name='ResumeBullet', fontName='Times-Roman', fontSize=10, leftIndent=18, spaceAfter=4, leading=12))
        styles.add(ParagraphStyle(name='SkillsStyle', fontName='Times-Roman', fontSize=10, leading=14))

        separator = HRFlowable(width="100%", thickness=0.7, color=colors.black, spaceBefore=4, spaceAfter=7)

        flowables = []

        # Header
        flowables.append(Paragraph(name, styles['NameStyle']))
        contact_info = f"{email} | {phone} | {linkedin}"
        flowables.append(Paragraph(contact_info, styles['ContactStyle']))
        #flowables.append(separator)

        # Professional Summary
        if professional_summary:
            flowables.append(Paragraph("Professional Summary", styles['SectionHead']))
            flowables.append(separator)
            flowables.append(Paragraph(professional_summary, styles['ResumeBullet']))
            flowables.append(Spacer(1, 4))

        # Education
        flowables.append(Paragraph("Education", styles['SectionHead']))
        flowables.append(separator)
        for edu in st.session_state.education_entries:
            if edu.get('degree') and edu.get('school'):
                left_text = f"<b>{edu['degree']}</b>, {edu['school']}"
                right_text = f"{edu['graduation_date']}"
                edu_table = Table(
                    [[Paragraph(left_text, styles['ResumeBullet']),
                    Paragraph(right_text, styles['JobDate'])]],
                    colWidths=[doc.width - 1.5*inch, 1.5*inch]  # adjust space
                )
                flowables.append(edu_table)
        flowables.append(Spacer(1, 4))

        # Work Experience
        flowables.append(Paragraph("Work Experience", styles['SectionHead']))
        flowables.append(separator)
        for exp in st.session_state.work_experiences:
            if exp.get('job_title') and exp.get('company'):
                header_data = [[Paragraph(f"{exp['job_title']}, {exp['company']}", styles['JobTitle']),
                                Paragraph(f"{exp['start_date']} - {exp['end_date']}", styles['JobDate'])]]
                header_table = Table(header_data, colWidths=[doc.width - 1.5*inch, 1.5*inch])
                flowables.append(header_table)
                for resp in exp['responsibilities'].splitlines():
                    if resp.strip():
                        flowables.append(Paragraph(resp.strip(), styles['ResumeBullet']))
                flowables.append(Spacer(1, 4))

        # Projects
        flowables.append(Paragraph("Projects", styles['SectionHead']))
        flowables.append(separator)
        for proj in st.session_state.project_entries:
            if proj.get('name'):
                proj_header = f"<b>{proj['name']}</b>"
                flowables.append(Paragraph(proj_header, styles['JobTitle']))
                if proj.get('technologies'):
                    flowables.append(Paragraph(f"<i>Technologies:</i> {proj['technologies']}", styles['ResumeBullet']))
                for desc in proj['description'].splitlines():
                    if desc.strip():
                        flowables.append(Paragraph(desc.strip(), styles['ResumeBullet']))
                flowables.append(Spacer(1, 4))

        # Skills
        flowables.append(Paragraph("Skills", styles['SectionHead']))
        flowables.append(separator)
        if skills:
           skills_text = ", ".join([s.strip() for s in skills.split(",") if s.strip()])
        flowables.append(Paragraph(skills_text, styles['SkillsStyle']))

        doc.build(flowables)

        buffer.seek(0)
        st.download_button(
            label="Download Resume as PDF",
            data=buffer,
            file_name=f"{name.replace(' ', '_')}_Resume.pdf",
            mime="application/pdf"
        )
        st.success("Resume PDF generated successfully!")
