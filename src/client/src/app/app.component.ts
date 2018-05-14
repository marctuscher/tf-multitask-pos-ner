import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
const httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
};
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
    title = 'app';
    predictions = [];
    sentence: string;

    constructor(private http: HttpClient){
    }
    ngOnInit(){
    }

    predict(){
        console.log(this.sentence)
        let body = JSON.stringify(this.sentence);
        this.http.post('http://localhost:8080/predict', body, httpOptions).subscribe(res => {
            console.log(res);
            this.sentence = ""
            this.predictions.unshift(res);
        })
    }
}



