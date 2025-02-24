run_app:
run_app:
	python3 dashboard.py & sleep 30
	wget -r -P pages_files http://127.0.0.1:8050/
	wget -r -P pages_files http://127.0.0.1:8050/_dash-layout
	wget -r -P pages_files http://127.0.0.1:8050/_dash-dependencies
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/dash/dcc/async-graph.js
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/dash/dcc/async-highlight.js
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/dash/dcc/async-markdown.js
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/dash/dcc/async-datepicker.js
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/dash/dash_table/async-table.js
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/dash/dash_table/async-highlight.js
	wget -r -P pages_files http://127.0.0.1:8050/_dash-component-suites/plotly/package_data/plotly.min.js
	find pages_files -type f -exec sed -i.bak 's|_dash-component-suites|tech-investment-analysis/_dash-component-suites|g' {} \;
	find pages_files -type f -exec sed -i.bak 's|_dash-layout|tech-investment-analysis/_dash-layout.json|g' {} \;
	find pages_files -type f -exec sed -i.bak 's|_dash-dependencies|tech-investment-analysis/_dash-dependencies.json|g' {} \;
	find pages_files -type f -exec sed -i.bak 's|_reload-hash|tech-investment-analysis/_reload-hash|g' {} \;
	find pages_files -type f -exec sed -i.bak 's|_dash-update-component|tech-investment-analysis/_dash-update-component|g' {} \;
	find pages_files -type f -exec sed -i.bak 's|assets|tech-investment-analysis/assets|g' {} \;
	mv pages_files/_dash-layout pages_files/_dash-layout.json || echo "_dash-layout not found"
	mv pages_files/_dash-dependencies pages_files/_dash-dependencies.json || echo "_dash-dependencies not found"
	mkdir -p pages_files/assets
	mv assets/* pages_files/assets/
	pkill -f 'python'

clean_dirs:
	rm -rf pages_files/
	rm -rf joblib
