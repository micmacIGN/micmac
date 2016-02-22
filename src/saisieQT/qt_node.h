#ifdef __DEBUG
	#if ELISE_windows
		#include <fstream>
		ofstream qt_out("qt_out.txt");
	#else
		#define qt_out cout
	#endif

	#include <typeinfo>
	#if ELISE_unix
		#include <cxxabi.h>
	#endif

	QWidget *mainWindow = NULL;

	class QT_node
	{
	public:
		QWidget *mValue;
		QT_node *mParent;
		list<QT_node *> mChildren;

	private:
		bool _addChild(QT_node &aChild)
		{
			QT_node * const child = &aChild;
			list<QT_node *>::iterator it = mChildren.begin();
			while (it != mChildren.end() && *it < child) it++;

			if (it == mChildren.end() && *it == child) return false;

			mChildren.insert(it, child);
			return true;
		}

	public:
		QT_node(QWidget *aValue):
			mValue(aValue),
			mParent(NULL)
		{
		}

		void setParent(QT_node &aParent)
		{
			ELISE_DEBUG_ERROR(mParent != NULL, "QT_node::setParent", "mParent != NULL");
			ELISE_DEBUG_ERROR(this == &aParent, "QT_node::setParent", "this == &aParent");

			mParent = &aParent;
			mParent->_addChild(*this);
		}

		bool isRoot() const { return mParent == NULL; }

		bool isLeaf() const { return mChildren.empty(); }

		void printDescent(const string &aPrefix, ostream &aStream)
		{
		
		}

		void getLeafs(list<QT_node *> &oLeafs)
		{
			oLeafs.clear();

			list<QT_node *> toProcess;
			toProcess.push_back(this);
			while ( !toProcess.empty())
			{
				QT_node &current = *toProcess.front();
				toProcess.pop_front();

				if (current.isLeaf())
					oLeafs.push_back(&current);
				else
					toProcess.insert(toProcess.begin(), current.mChildren.begin(), current.mChildren.end());
			}
		}

		bool hasChild(const QT_node &aNode) const
		{
			list<QT_node *>::const_iterator it = mChildren.begin();
			while (it != mChildren.end())
				if (*it++ == &aNode) return true;
			return false;
		}

		size_t height() const
		{
			if (isLeaf()) return 0;

			list<QT_node *>::const_iterator it = mChildren.begin();
			size_t maxHeight = 0;
			while (it != mChildren.end())
			{
				size_t height = (**it++).height();
				if (height > maxHeight) maxHeight = height;
			}

			return maxHeight + 1;
		}

		size_t depth() const
		{
			size_t result = 0;
			const QT_node *node = this;
			while (node->mParent != NULL)
			{
				result++;
				node = node->mParent;
			}
			return result;
		}

		string name() const
		{
			stringstream ss;

			#if ELISE_unix
				int status;
				string name = abi::__cxa_demangle(typeid(*mValue).name(), NULL, NULL, &status);
			#else
				string name = typeid(*mValue).name();
			#endif

			ss << name << '(' << mValue->objectName().toStdString() << ',' << mValue->actions().size() << " actions)";

			//~ QAction *help = NULL;
			foreach (QAction *action, mValue->actions())
			{
				//~ if (mainWindow != NULL) mainWindow->addAction(action);

				//~ ss << ' ' << action->objectName().toStdString();
				ss << " [" << gAllActions[action] << ']';
//~ 
				//~ if (action->objectName() == "actionHelpShortcuts") help = action;
			}
			//~ ss << ')';
//~ 
			//~ if (help != NULL)
			//~ {
				//~ qt_out << "action help found !" << endl;
				//~ help->trigger();
			//~ }

			return ss.str();
		}

		string lineage() const
		{
			stringstream ss;
			list<const QT_node *>lineage;
			const QT_node *node = this;
			while ( !node->isRoot())
			{
				lineage.push_front(node);
				node = node->mParent;
			}
			lineage.push_front(node);

			list<const QT_node *>::const_iterator it = lineage.begin();
			ss << (**it++).name();
			while (it != lineage.end())
				ss << " <- " << (**it++).name();

			return ss.str();
		}

		#ifdef __DEBUG
			void __check_connections() const
			{
				ELISE_DEBUG_ERROR(mParent != NULL && !mParent->hasChild(*this), "__check_connections", "node's parent has not node has a child");
			}
		#endif
	};

	class QT_forest
	{
	public:
		list<QT_node> mNodes;

	private:
		bool _addLineage(QT_node &aNode)
		{
			bool result = false;
			QT_node *node = &aNode, *parentNode = NULL;
			QWidget *parent = node->mValue->parentWidget();
			while (node->mParent == NULL && parent != NULL)
			{
				if (_add(parent, &parentNode)) result = true;
				if (node->mParent == NULL) node->setParent(*parentNode);
				node = parentNode;
				parent = node->mValue->parentWidget();
			}
			return result;
		}

		bool _add(QWidget *aValue, QT_node **oAddedNode = NULL)
		{
			ELISE_DEBUG_ERROR(aValue == NULL, "QT_forest::add", "aValue == NULL");

			list<QT_node>::iterator it = mNodes.begin();
			while (it != mNodes.end() && it->mValue < aValue) it++;

			bool result = false;
			if (it == mNodes.end() || it->mValue != aValue)
			{
				it = mNodes.insert(it, QT_node(aValue));

				ELISE_DEBUG_ERROR(it->mValue != aValue, "QT_forest::add", "it->mValue != aValue");

				result = true;
			}

			if (oAddedNode != NULL) *oAddedNode = &*it;
			return result;
		}

	public:
		void addLineage(QWidget *aValue)
		{
			ELISE_DEBUG_ERROR(aValue == NULL, "QT_forest::addLineage", "aValue == NULL");

			QT_node *node;
			if (_add(aValue, &node)) _addLineage(*node);
		}

		size_t nbRoots() const
		{
			size_t result = 0;
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
				if ((*it++).isRoot()) result++;
			return result;
		}

		size_t nbLeafs() const
		{
			size_t result = 0;
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
				if ((*it++).isLeaf()) result++;
			return result;
		}

		void print(const string &aPrefix, ostream &aStream)
		{
		}

		size_t maxHeight() const
		{
			size_t result = 0;
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
			{
				if (it->isRoot()) result = max<size_t>(result, it->height());
				it++;
			}
			return result;
		}

		size_t minHeight() const
		{
			size_t result = numeric_limits<size_t>::max();
			list<QT_node>::const_iterator it = mNodes.begin();
			while (it != mNodes.end())
			{
				if (it->isRoot()) result = min<size_t>(result, it->height());
				it++;
			}
			return result;
		}

		void getRoots(list<QT_node *> &oRoots)
		{
			oRoots.clear();

			list<QT_node>::iterator it = mNodes.begin();
			while (it != mNodes.end())
			{
				if (it->isRoot()) oRoots.push_back(&*it);
				it++;
			}
		}

		static QWidget *getRoot(QWidget *aWidget)
		{
			QWidget *widget = NULL;
			while (aWidget != NULL)
			{
				widget = aWidget;
				aWidget = aWidget->parentWidget();
			}
			return widget;
		}

		#ifdef __DEBUG
			void __check_connections() const
			{
				list<QT_node>::const_iterator it = mNodes.begin();
				while (it != mNodes.end()) (*it++).__check_connections();
			}
		#endif
	};
#endif
