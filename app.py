import os
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session
)
from werkzeug.utils import secure_filename
from data_utils import save_uploaded_file
from ai_handler import analyze_data, generate_visualization


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
SECRET_KEY = 'change_this_to_a_secure_random_value'


app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/reset')
def reset():
    session.pop('csv_path', None)
    flash('🔄 Session reset. Please upload a new CSV.', 'info')
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    # Upload handling
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or file.filename == '':
            flash('No CSV file selected.', 'danger')
            return redirect(url_for('index'))

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a CSV.', 'danger')
            return redirect(url_for('index'))

        # Save and store path in session
        filename = secure_filename(file.filename)
        filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        session['csv_path'] = filepath

       # flash('✅ Data uploaded successfully!', 'success')
        return redirect(url_for('index'))  # PRG: prevent flash duplication

    # GET request: check if CSV loaded
    csv_loaded = 'csv_path' in session and os.path.exists(session['csv_path'])
    table_preview = None
    if csv_loaded:
        df = pd.read_csv(session['csv_path'],encoding='ISO-8859-1')
        table_preview = df.head(3).to_html(
            classes="table table-striped table-bordered",
            index=False
        )

    return render_template(
        'index.html',
        csv_loaded=csv_loaded,
        table_preview=table_preview
    )


@app.route('/visualize', methods=['POST'])
def visualize():
    if 'csv_path' not in session or not os.path.exists(session['csv_path']):
        flash('Please upload a CSV file first.', 'warning')
        return redirect(url_for('index'))

    query = request.form.get('vis_query', '').strip()
    if not query:
        flash('Please enter a visualization request.', 'warning')
        return redirect(url_for('index'))

   
    df = pd.read_csv(session['csv_path'])
    image_filename = generate_visualization(df, query)
    if image_filename:
        image_url = url_for('static', filename=image_filename)
        flash('🖼️ Visualization generated!', 'success')
    else:
        image_url = None
        flash('Failed to generate visualization.', 'danger')


    table_preview = df.head(3).to_html(
        classes="table table-striped table-bordered",
        index=False
    )
    return render_template(
        'index.html',
        csv_loaded=True,
        table_preview=table_preview,
        image_path=image_url,
        vis_query=query,
        active_tab='visualize'
    )


@app.route('/ask', methods=['POST'])
def ask():
    if 'csv_path' not in session or not os.path.exists(session['csv_path']):
        flash('Please upload a CSV file first.', 'warning')
        return redirect(url_for('index'))

    query = request.form.get('qa_query', '').strip()
    if not query:
        flash('Please enter a question.', 'warning')
        return redirect(url_for('index'))


    df = pd.read_csv(session['csv_path'], encoding='ISO-8859-1')
    response = analyze_data(df, query, mode='qa')
    print('Response',response)
    print('Test App')

    import re
    answer = re.search(r'Answer:\s*(.*?)(?=\n|$)', response, re.IGNORECASE)
    print('Answer',answer)
    #print(answer.group(1))
    reason = re.search(r'Reason:\s*(.*?)(?=\n|$)', response, re.IGNORECASE)
    stats = re.search(r'Stats:\s*(.*?)(?=\n|$)', response, re.IGNORECASE)

    table_preview = df.head(3).to_html(
        classes="table table-striped table-bordered",
        index=False
    )

    flash('💬 Analysis complete!', 'success')
    return render_template(
        'index.html',
        csv_loaded=True,
        table_preview=table_preview,
        answer=answer.group(1) if answer else None,
        reason=reason.group(1) if reason else None,
        stats=stats.group(1) if stats else None,
        qa_query=query,
        active_tab='ask'
    )


if __name__ == '__main__':
    app.run(debug=True)
