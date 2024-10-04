from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
