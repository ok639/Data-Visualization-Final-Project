app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <title>Dash App</title>
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script>
                document.getElementById('save-dates-btn').onclick = function() {
                    var startDate = document.getElementById('start-date-input').value;
                    var endDate = document.getElementById('end-date-input').value;
                    sessionStorage.setItem('start_date', startDate);
                    sessionStorage.setItem('end_date', endDate);
                };
                window.onload = function() {
                    var storedStartDate = sessionStorage.getItem('start_date');
                    var storedEndDate = sessionStorage.getItem('end_date');
                    if(storedStartDate && storedEndDate) {
                        document.getElementById('start-date-input').value = storedStartDate;
                        document.getElementById('end-date-input').value = storedEndDate;
                    }
                };
            </script>
        </footer>
    </body>
</html>
"""
