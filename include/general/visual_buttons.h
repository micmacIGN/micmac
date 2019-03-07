#ifndef VISUAL_BUTTONS_H
#define VISUAL_BUTTONS_H

#include "Elise_QT.h"
#include <cmath>

class cSelectionButton: public QPushButton
{
    Q_OBJECT

public:
        cSelectionButton(QString text, int id = -1, bool normalize = false, QWidget *parent=0);

protected:
        int  _m_line_id;
        bool _m_nrm;

public slots:
        void onClick();

signals:
        void my_click(int, bool);
        void my_click(int);

};

//SpinBox showing power of 2
class cSpinBox: public QSpinBox
{
    Q_OBJECT

public:
        cSpinBox(int value, QWidget *parent=0);

        void stepBy(int steps);

private:
        QVector <int> _m_values;
        int           _m_index;

};
#endif // VISUAL_BUTTONS_H