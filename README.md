# RecyclingPics

A tool to take photos of recycling, and have it categorise the items dependent upon which bin they need to go in

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

#run with: python -m src.RecyclingPics.main

#general function: load image, run model, classify items, output bin suggestion