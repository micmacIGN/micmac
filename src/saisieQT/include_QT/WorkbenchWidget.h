#ifndef WORKBENCHWIDGET_H
#define WORKBENCHWIDGET_H

#include <QWidget>

namespace Ui {
class cWorkBenchWidget;
}

class cWorkBenchWidget : public QWidget
{
	Q_OBJECT

public:
	explicit cWorkBenchWidget(QWidget *parent = 0);
	~cWorkBenchWidget();

private:
	Ui::cWorkBenchWidget *ui;
};

#endif // WORKBENCHWIDGET_H
