package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/julienschmidt/httprouter"
)

// APIResponse our generic response object to return JSON to the end user
type APIResponse struct {

	//required fields
	Status  string `json:"status"`
	Message string `json:"message"`
	Code    int64  `json:"code"`

	//optional fields
	ID int64 `json:"id,omitempty"`
}

func responseToJSON(s APIResponse) string {
	//take in a API response struct and then convert it to json and return the json string
	b, _ := json.Marshal(s)
	str := string(b)
	return str
}

func lookup(resp http.ResponseWriter, req *http.Request, params httprouter.Params) {
	resp.Header().Set("Content-Type", "application/json")
	respMessage := APIResponse{Status: "success", Message: "Found " + params.ByName("id"), Code: 0}
	fmt.Fprintf(resp, responseToJSON(respMessage))
}

func create(resp http.ResponseWriter, req *http.Request, params httprouter.Params) {
	resp.Header().Set("Content-Type", "application/json")
	respMessage := APIResponse{}
	if req.FormValue("user") == "" {
		respMessage = APIResponse{Status: "error", Message: "user is required.", Code: 100}
	} else {
		respMessage = APIResponse{Status: "success", Message: "Created " + req.FormValue("user"), ID: 43564, Code: 4000}
	}
	fmt.Fprintf(resp, responseToJSON(respMessage))
}

func demo() {
	server := httprouter.New()
	server.GET("/:id", lookup)
	server.POST("/", create)
	http.ListenAndServe(":80", server)
}
