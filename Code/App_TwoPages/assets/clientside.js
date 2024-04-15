if (!window.dash_clientside) { window.dash_clientside = {}; }

window.dash_clientside.clientside = {
    store_dates: function(n_clicks, start_date, end_date) {
        if(n_clicks > 0) {
            sessionStorage.setItem('start_date', start_date);
            sessionStorage.setItem('end_date', end_date);
            return {start: start_date, end: end_date};  // Save to Dash Store component
        }
        return window.dash_clientside.no_update; // No update if not clicked
    },
    load_dates: function(timestamp, data) {
        if (timestamp) {
            return [data.start, data.end];
        }
        return [null, null];  // Return null if no data available
    }
};