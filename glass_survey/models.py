from . import db

class Pointing(db.Model):
    source = db.Column(db.String(9), primary_key=True)
    ra = db.Column(db.Float, nullable=False)
    dec = db.Column(db.Float, nullable=False)
    field = db.Column(db.String(1))
    daily_set = db.Column(db.Integer)
    group = db.Column(db.String(1))

    def __repr__(self):
        return f"<Pointing {self.source}>"


class Scan(db.Model):
    datetime = db.Column(db.DateTime, primary_key=True)
    lmst = db.Column(db.Float)
    n_unflagged = db.Column(db.Integer)
    frac_vis = db.Column(db.Float)
    array_config = db.Column(db.String(4))

    source_id = db.Column(db.String(9), db.ForeignKey("pointing.source"), nullable=False)
    source = db.relationship('Pointing', backref=db.backref('scans', lazy=True))

    def __repr__(self):
        return f"<Scan {self.source} at {self.datetime}>"
